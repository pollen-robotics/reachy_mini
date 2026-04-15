"""Cross-thread concurrency lock for the robot.

The daemon is the single source of truth for "is this robot in use?".
Two independent code paths can try to drive the robot:

- Local Python apps launched by AppManager (in the main asyncio loop).
- Remote WebRTC clients handled by the central signaling relay (in the
  relay's dedicated thread + event loop).

Without coordination, both paths could run simultaneously and fight over
motors/media. This lock serializes them.

States are mutually exclusive:

- ``free``: nobody holds the robot.
- ``local_app(name)``: a Python app is running.
- ``remote_session(name)``: a remote client is connected via central.

Transitions are triggered by:

- ``AppManager.start_app`` / app exit  → local_app / free
- Relay receives ``startSession`` / ``endSession`` / disconnect → remote_session / free

A local-app acquire evicts any remote session in progress: the remote
peer is notified via ``endSession`` so it can tear down cleanly, and the
relay stops accepting new remote sessions until the local app exits.
A remote-session acquire is refused outright if a local app is running.
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Awaitable, Callable, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RobotLockState(str, Enum):
    """Lock state enum. Values are stable strings suitable for serialization."""

    FREE = "free"
    LOCAL_APP = "local_app"
    REMOTE_SESSION = "remote_session"


class RobotLockStatus(BaseModel):
    """Snapshot of the lock state, suitable for JSON serialization.

    Returned by :meth:`RobotLock.status` and by the
    ``GET /api/daemon/robot-lock-status`` endpoint.
    """

    state: RobotLockState
    holder_name: Optional[str] = None


class RobotLock:
    """Thread-safe lock coordinating local app and remote session access to the robot."""

    def __init__(self) -> None:
        """Initialize the lock in the free state."""
        self._mutex = threading.Lock()
        self._state: RobotLockState = RobotLockState.FREE
        self._holder_name: Optional[str] = None

        # Async callback invoked when a local-app acquire evicts a remote
        # session. Registered by the relay via ``set_remote_eviction_handler``.
        # Called *outside* the mutex to avoid blocking other acquirers while
        # the relay tears down its sessions.
        #
        # Thread-safety note: this attribute is written only by start()/stop()
        # on the relay (which run on the main asyncio loop) and read by
        # acquire_local_evicting_remote (also main asyncio loop, invoked from
        # AppManager). Because both writer and reader share one thread,
        # Python's GIL makes the bare reference assignment safe without a
        # mutex. If you ever invoke set_remote_eviction_handler from a
        # different thread, add a mutex here.
        self._on_remote_evicted: Optional[Callable[[], Awaitable[None]]] = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def set_remote_eviction_handler(
        self, handler: Optional[Callable[[], Awaitable[None]]]
    ) -> None:
        """Register (or clear) the coroutine invoked when a local acquire evicts a remote session.

        The handler must be safe to call from the caller of
        ``acquire_local_evicting_remote`` — typically the main asyncio loop.
        Pass ``None`` to clear.
        """
        self._on_remote_evicted = handler

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def status(self) -> RobotLockStatus:
        """Return a snapshot of the current lock state."""
        with self._mutex:
            return RobotLockStatus(state=self._state, holder_name=self._holder_name)

    # ------------------------------------------------------------------
    # Local acquire / release
    # ------------------------------------------------------------------

    async def acquire_local_evicting_remote(self, app_name: str) -> None:
        """Acquire the lock for a local Python app, evicting any remote session.

        Raises:
            RuntimeError: If another local app already holds the lock. Caller
                should not start a second Python app concurrently.

        If a remote session is held, it is transitioned to ``local_app``
        atomically and the registered eviction handler is invoked (after
        releasing the mutex) so the relay can notify the remote peer.

        """
        evict_remote = False
        with self._mutex:
            if self._state == RobotLockState.LOCAL_APP:
                raise RuntimeError(
                    f"A local app is already running: {self._holder_name!r}"
                )
            if self._state == RobotLockState.REMOTE_SESSION:
                evict_remote = True
                logger.info(
                    "RobotLock: evicting remote session %r for local app %r",
                    self._holder_name,
                    app_name,
                )
            self._state = RobotLockState.LOCAL_APP
            self._holder_name = app_name
            logger.info("RobotLock: acquired by local app %r", app_name)

        if evict_remote and self._on_remote_evicted is not None:
            # Invoke outside the mutex. The handler schedules tear-down on
            # the relay thread; it must not attempt to acquire the lock
            # again (would deadlock if it did).
            try:
                await self._on_remote_evicted()
            except Exception:
                # Tear-down is best-effort: the state machine has already
                # moved to local_app, so new remote sessions will be
                # rejected regardless. We log and continue.
                logger.warning(
                    "RobotLock: remote eviction handler raised", exc_info=True
                )

    def release_local(self, app_name: Optional[str] = None) -> None:
        """Release the lock held by a local app.

        Idempotent: if the lock is free or held by a remote session, this
        is a no-op (with a warning). Safe to call from ``monitor_process``
        regardless of how the subprocess exited.

        Args:
            app_name: Optional name of the app expected to hold the lock.
                If provided and the current holder differs, logs a warning
                but still releases — this protects against stale releases
                after a rapid stop/start cycle.

        """
        with self._mutex:
            if self._state != RobotLockState.LOCAL_APP:
                logger.debug(
                    "RobotLock.release_local: not holding local_app (state=%s)",
                    self._state.value,
                )
                return
            if app_name is not None and self._holder_name != app_name:
                logger.warning(
                    "RobotLock.release_local: holder mismatch (expected=%r holder=%r); releasing anyway",
                    app_name,
                    self._holder_name,
                )
            released_name = self._holder_name
            self._state = RobotLockState.FREE
            self._holder_name = None
            logger.info("RobotLock: released by local app %r", released_name)

    # ------------------------------------------------------------------
    # Remote acquire / release
    # ------------------------------------------------------------------

    def try_acquire_remote(self, app_name: str) -> bool:
        """Attempt to acquire the lock for a remote WebRTC session.

        Returns:
            True if the lock was acquired (state transitioned to
            ``remote_session``). False if a local app or another remote
            session is already holding it — caller must refuse the
            incoming session.

        """
        with self._mutex:
            if self._state != RobotLockState.FREE:
                logger.info(
                    "RobotLock: remote acquire refused (state=%s holder=%r requester=%r)",
                    self._state.value,
                    self._holder_name,
                    app_name,
                )
                return False
            self._state = RobotLockState.REMOTE_SESSION
            self._holder_name = app_name
            logger.info("RobotLock: acquired by remote session %r", app_name)
            return True

    def release_remote(self) -> None:
        """Release a remote-session hold. Idempotent."""
        with self._mutex:
            if self._state != RobotLockState.REMOTE_SESSION:
                logger.debug(
                    "RobotLock.release_remote: not holding remote_session (state=%s)",
                    self._state.value,
                )
                return
            released_name = self._holder_name
            self._state = RobotLockState.FREE
            self._holder_name = None
            logger.info("RobotLock: released by remote session %r", released_name)
