"""Central signaling relay for WebRTC.

Connects to a central signaling server via HTTP/SSE and relays messages to/from
the local GStreamer webrtcsink signaling server.

The relay automatically:
- Reconnects on connection failures
- Refreshes the HF token on reconnection attempts
- Responds to token updates (login/logout) without restart
"""

import asyncio
import json
import logging
import os
import threading
from enum import Enum
from typing import Any, Callable, Optional

import aiohttp
import websockets
from websockets.asyncio.client import ClientConnection

from reachy_mini.daemon.robot_app_lock import RobotAppLock
from reachy_mini.utils.hardware_id import get_hardware_id

logger = logging.getLogger(__name__)


class RelayState(Enum):
    """Connection state of the central signaling relay."""

    STOPPED = "stopped"
    WAITING_FOR_TOKEN = "waiting_for_token"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


# Central signaling server URL. Override via the REACHY_CENTRAL_URL env
# var at startup to point at a fork (test Space, staging, etc.). The
# canonical is the org-owned Space; the legacy cduss / tfrere instances
# stay running for backward compatibility during the transition.
CENTRAL_SIGNALING_SERVER = os.getenv(
    "REACHY_CENTRAL_URL", "https://pollen-robotics-reachy-mini-central.hf.space"
)
LOCAL_GSTREAMER_SIGNALING = "ws://127.0.0.1:8443"

# Reconnection settings
RECONNECT_INTERVAL = 5.0  # seconds
TOKEN_CHECK_INTERVAL = 30.0  # seconds - how often to check for token when not connected
LOCAL_WS_CONNECT_TIMEOUT = 5.0  # seconds - timeout for local websocket connection
LOCAL_WS_WELCOME_TIMEOUT = (
    3.0  # seconds - timeout waiting for welcome message from local
)
SSE_READ_TIMEOUT = (
    60.0  # seconds - timeout for reading from SSE stream (should receive keepalive)
)

# Producer-listed health check (self-heal split-brain states where the
# relay's SSE is alive but central no longer lists this robot as a
# producer for the authenticated user, e.g. because a setPeerStatus
# round-trip was cancelled mid-flight).
PRODUCER_HEALTH_CHECK_INTERVAL = 30.0  # seconds - poll cadence
PRODUCER_HEALTH_CHECK_INITIAL_DELAY = (
    10.0  # seconds - wait after welcome before the first poll
)
PRODUCER_HEALTH_CHECK_TIMEOUT = 10.0  # seconds - per-poll HTTP timeout
PRODUCER_HEALTH_CHECK_MAX_MISSES = (
    2  # consecutive missing-from-list polls before we self-trigger a force_reconnect
)

# Heartbeat: re-emit setPeerStatus periodically so central refreshes our
# peer lease (it runs a TTL sweeper that evicts producers without recent
# inbound traffic). Cadence is negotiated through the SSE welcome; these
# fall back when central does not advertise one. Details in `_heartbeat_loop`.
HEARTBEAT_DEFAULT_INTERVAL = 5.0  # fallback when central does not advertise
HEARTBEAT_MIN_INTERVAL = 1.0  # sanity floor (prevents request storms)
HEARTBEAT_MAX_INTERVAL = 60.0  # sanity ceiling


def _clamp_heartbeat_interval(value: float) -> float:
    """Clamp a candidate heartbeat interval to the [MIN, MAX] safety envelope."""
    return max(HEARTBEAT_MIN_INTERVAL, min(HEARTBEAT_MAX_INTERVAL, value))


class CentralSignalingRelay:
    """Relay signaling messages between central server (HTTP/SSE) and local GStreamer (WebSocket).

    This class maintains connections to both a central signaling server (for remote access)
    and the local GStreamer WebRTC signaling server, relaying messages between them.

    The relay is designed to be robust:
    - Automatically reconnects on connection failures
    - Refreshes the HF token on each reconnection attempt
    - Can be notified of token changes for immediate reconnection
    """

    def __init__(
        self,
        central_uri: str = CENTRAL_SIGNALING_SERVER,
        local_uri: str = LOCAL_GSTREAMER_SIGNALING,
        hf_token: Optional[str] = None,
        robot_name: str = "reachymini",
        transport: str = "wifi",
        on_state_change: Optional[Callable[["RelayState", Optional[str]], None]] = None,
        robot_app_lock: Optional[RobotAppLock] = None,
    ):
        """Initialize the relay.

        Args:
            central_uri: HTTP URI of central signaling server
            local_uri: WebSocket URI of local GStreamer signaling server
            hf_token: HuggingFace token for authentication (will be refreshed)
            robot_name: Name to register as producer
            transport: How a client physically reaches this daemon,
                advertised as ``meta.transport`` and forwarded verbatim
                by central to listeners. ``"wifi"`` for an autonomous
                Pi-side daemon (Wireless variant or any Lite with its
                own network), ``"usb"`` for the desktop-tray daemon
                spawned next to the user's machine and tethered to the
                robot over USB. The mobile picker uses this to badge
                cards "USB" vs "Wi-Fi" without round-tripping the daemon.
                Free-form string so future fronts (``"ethernet"``,
                ``"sim"``, ...) need no relay change.
            on_state_change: Callback when state changes (state, message)
            robot_app_lock: Shared lock coordinating local vs remote access to
                the robot. When provided, incoming remote sessions are
                gated on this lock and a local-app acquire evicts any
                active remote session.

        """
        self.central_uri = central_uri
        self.local_uri = local_uri
        self.hf_token = hf_token
        self.robot_name = robot_name
        self.transport = transport
        self._on_state_change = on_state_change
        self._robot_app_lock = robot_app_lock

        self._running = False
        self._state = RelayState.STOPPED
        self._state_message: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._thread_loop: Optional["asyncio.AbstractEventLoop"] = None
        self._local_ws: Optional[ClientConnection] = None
        self._central_peer_id: Optional[str] = None
        self._local_peer_id: Optional[str] = None
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._connection_attempts = 0
        # Reentrancy guard for `_close_connections`. A teardown can be
        # triggered from three converging paths on the same event loop:
        #   1. `_reconnect()` scheduled by `force_reconnect()` /
        #      `update_token()` (via run_coroutine_threadsafe);
        #   2. `_watch_for_token_update` woken by `_token_updated.set()`
        #      (which the same `_reconnect()` set);
        #   3. `_connect_and_relay`'s `finally:` block, fired when
        #      `_producer_health_loop` ends after its self-triggered
        #      `force_reconnect()` and wins the FIRST_COMPLETED race.
        # All three call `_close_connections()`. The function used to
        # be idempotent only by accident (each `await close()` happens
        # to swallow the "already closed" error); making the invariant
        # explicit so future state cleanup additions can't regress it.
        self._closing = False

        # Event to signal token update (triggers immediate reconnection)
        self._token_updated = asyncio.Event()

        # Map central session IDs to client peer IDs
        self._session_to_local_peer: dict[str, str] = {}
        self._local_producer_id: Optional[str] = None

        # Session ID mapping between central and local
        self._pending_central_sessions: list[
            str
        ] = []  # Central sessions waiting for local session
        self._local_to_central_session: dict[
            str, str
        ] = {}  # local_session_id -> central_session_id
        self._central_to_local_session: dict[
            str, str
        ] = {}  # central_session_id -> local_session_id

        # Cadence at which `_heartbeat_loop` re-emits setPeerStatus
        # to refresh central's peer lease. Negotiated from the SSE
        # `welcome` message every reconnect; falls back to
        # HEARTBEAT_DEFAULT_INTERVAL if central does not advertise
        # one (pre-lifecycle-robustness deployments).
        self._heartbeat_interval_seconds: float = HEARTBEAT_DEFAULT_INTERVAL

    @property
    def state(self) -> RelayState:
        """Get the current connection state."""
        return self._state

    @property
    def state_message(self) -> Optional[str]:
        """Get additional info about the current state."""
        return self._state_message

    def _set_state(self, state: RelayState, message: Optional[str] = None) -> None:
        """Update the connection state with logging."""
        old_state = self._state
        if old_state == state and self._state_message == message:
            return

        self._state = state
        self._state_message = message

        # Log state transition with appropriate level
        log_msg = (
            f"[Central Relay] State transition: {old_state.value} -> {state.value}"
        )
        if message:
            log_msg += f" | {message}"

        if state == RelayState.CONNECTED:
            logger.info(log_msg)
        elif state == RelayState.ERROR:
            logger.warning(log_msg)
        elif state == RelayState.WAITING_FOR_TOKEN:
            logger.info(log_msg)
        elif state == RelayState.RECONNECTING:
            logger.info(log_msg)
        elif state == RelayState.CONNECTING:
            logger.debug(log_msg)
        elif state == RelayState.STOPPED:
            logger.info(log_msg)
        else:
            logger.debug(log_msg)

        # Notify callback if set
        if self._on_state_change:
            try:
                self._on_state_change(state, message)
            except Exception as e:
                logger.debug(f"[Central Relay] State change callback error: {e}")

    async def start(self) -> None:
        """Start the relay service."""
        if self._running:
            logger.debug("[Central Relay] start() called but already running")
            return

        logger.info("[Central Relay] Starting relay service...")
        self._running = True
        self._connection_attempts = 0
        self._token_updated.clear()
        self._set_state(RelayState.CONNECTING, "Starting relay service...")

        # Register ourselves as the remote-eviction handler for the lock.
        # When AppManager acquires the lock for a local Python app, this
        # coroutine is invoked on the main asyncio loop and schedules the
        # actual tear-down on the relay's own thread loop.
        if self._robot_app_lock is not None:
            self._robot_app_lock.set_remote_eviction_handler(
                self._handle_remote_eviction
            )

        # Run the relay in its own thread with a dedicated event loop.
        # This is necessary because the caller (daemon.start) may run in a temporary
        # event loop that gets destroyed when the HTTP request handler completes.
        self._thread = threading.Thread(target=self._run_in_thread, daemon=True)
        self._thread.start()
        logger.info(f"[Central Relay] Relay thread started: {self._thread.name}")

    async def _handle_remote_eviction(self) -> None:
        """Cross-thread entry point for the lock's remote-eviction callback.

        Runs on the main asyncio loop (caller of ``acquire_local_evicting_remote``)
        but dispatches the actual work onto the relay's thread loop.
        """
        if self._thread_loop is None or not self._thread_loop.is_running():
            logger.debug(
                "[Central Relay] Eviction requested but relay loop not running; nothing to do"
            )
            return

        fut = asyncio.run_coroutine_threadsafe(
            self._tear_down_active_sessions(reason="local_app_started"),
            self._thread_loop,
        )
        # Wait for the tear-down to complete so AppManager knows the remote
        # peer has been notified before the local app starts up.
        try:
            await asyncio.wrap_future(fut)
        except Exception:
            logger.warning(
                "[Central Relay] Remote eviction tear-down raised", exc_info=True
            )

    async def _tear_down_active_sessions(self, reason: str) -> None:
        """End every active/pending remote session and notify both sides.

        Runs on the relay's thread event loop.
        """
        # Notify central so it clears its own session_id on the producer
        # and sends endSession to the remote consumer.
        for central_session_id in list(self._central_to_local_session.keys()):
            logger.info(
                "[Central Relay] Tearing down central session %s (reason=%s)",
                central_session_id,
                reason,
            )
            try:
                await self._send_to_central(
                    {
                        "type": "endSession",
                        "sessionId": central_session_id,
                        "reason": reason,
                    }
                )
            except Exception:
                logger.warning(
                    "[Central Relay] Failed to notify central of session teardown",
                    exc_info=True,
                )

        # Notify local GStreamer so it closes its RTCPeerConnection.
        for local_session_id in list(self._local_to_central_session.keys()):
            try:
                await self._send_to_local(
                    {"type": "endSession", "sessionId": local_session_id}
                )
            except Exception:
                logger.warning(
                    "[Central Relay] Failed to notify local of session teardown",
                    exc_info=True,
                )

        # Clear all session bookkeeping.
        self._pending_central_sessions.clear()
        self._local_to_central_session.clear()
        self._central_to_local_session.clear()
        self._session_to_local_peer.clear()

    def _run_in_thread(self) -> None:
        """Run the relay loop in a dedicated thread with its own event loop."""
        logger.info("[Central Relay] Thread starting, creating event loop...")
        self._thread_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._thread_loop)
        try:
            self._thread_loop.run_until_complete(self._run_loop())
        except Exception as e:
            logger.error(f"[Central Relay] Thread event loop error: {e}")
        finally:
            logger.info("[Central Relay] Thread event loop finished")
            self._thread_loop.close()
            self._thread_loop = None

    async def stop(self) -> None:
        """Stop the relay service."""
        logger.info("[Central Relay] Stopping relay service...")
        self._running = False

        # Wake up any waiting in the thread's event loop
        if self._thread_loop and self._thread_loop.is_running():
            self._thread_loop.call_soon_threadsafe(self._token_updated.set)

        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("[Central Relay] Thread did not stop within timeout")

        self._thread = None

        # Unregister from the lock and release any hold we may still have.
        if self._robot_app_lock is not None:
            self._robot_app_lock.set_remote_eviction_handler(None)
            self._robot_app_lock.release_remote()

        self._set_state(RelayState.STOPPED, "Relay stopped")

    async def update_token(self, new_token: Optional[str]) -> None:
        """Update the HF token and trigger reconnection if needed.

        This method should be called when the user logs in or out of
        HuggingFace. It will:
        - Update the stored token
        - Close existing connections
        - Trigger immediate reconnection attempt

        Args:
            new_token: The new HF token, or None if logged out

        """
        old_token = self.hf_token
        self.hf_token = new_token

        if old_token == new_token:
            logger.debug("[Central Relay] Token unchanged, no action needed")
            return

        await self._reconnect_now(new_token, reason="HF token updated, reconnecting...")

    async def force_reconnect(self) -> None:
        """Drop the current connection and reconnect with the stored token.

        Unlike ``update_token``, this path is NOT guarded by a token
        equality check - it always tears down the SSE and reconnects.

        Intended as a recovery handle for split-brain states where the
        relay thinks it is connected but central no longer lists this
        robot as a producer (see ``POST /api/hf-auth/refresh-relay``).
        """
        await self._reconnect_now(self.hf_token, reason="Forced relay reconnect")

    async def _reconnect_now(self, token: Optional[str], reason: str) -> None:
        """Shared core of token-change and force-reconnect paths.

        Transitions the relay into the right state and signals the run
        loop to tear down the current connection and try connecting
        again. Safe to call from any thread - if we have a running
        thread loop we schedule the close/set there, otherwise we set
        the event directly (covers the case where the relay has not
        started its background thread yet).
        """
        if token:
            self._set_state(RelayState.RECONNECTING, reason)
            self._connection_attempts = 0
        else:
            self._set_state(RelayState.WAITING_FOR_TOKEN, "Logged out from HuggingFace")

        if self._thread_loop and self._thread_loop.is_running():

            async def _reconnect() -> None:
                await self._close_connections()
                self._token_updated.set()

            asyncio.run_coroutine_threadsafe(_reconnect(), self._thread_loop)
        else:
            self._token_updated.set()

    def _refresh_token(self) -> Optional[str]:
        """Refresh the HF token from huggingface_hub.

        Returns:
            The current HF token, or None if not available

        """
        try:
            from huggingface_hub import get_token

            token = get_token()
            if token != self.hf_token:
                if token:
                    logger.info("[Central Relay] HF token detected (user logged in)")
                else:
                    logger.debug("[Central Relay] No HF token available")
                self.hf_token = token
            return token
        except Exception as e:
            logger.debug(f"[Central Relay] Could not get HF token: {e}")
            return self.hf_token

    async def _close_connections(self) -> None:
        """Close all connections.

        Reentrant-safe: three converging paths can call this on the
        same event loop in quick succession (see ``self._closing`` in
        ``__init__`` for the full triad). The guard short-circuits the
        second and third calls so a single teardown can't double-close
        an already-closing aiohttp session or interleave halfway
        through the dict clears.
        """
        if self._closing:
            return
        self._closing = True
        try:
            if self._http_session:
                try:
                    await self._http_session.close()
                except Exception:
                    pass
                self._http_session = None

            if self._local_ws:
                try:
                    await self._local_ws.close()
                except Exception:
                    pass
                self._local_ws = None

            # Clear session state
            self._central_peer_id = None
            self._local_peer_id = None
            self._session_to_local_peer.clear()
            self._pending_central_sessions.clear()
            self._local_to_central_session.clear()
            self._central_to_local_session.clear()

            # Release any remote hold on the robot lock. Idempotent: no-op
            # if we weren't holding it (e.g. local app currently has the
            # robot).
            if self._robot_app_lock is not None:
                self._robot_app_lock.release_remote()
        finally:
            self._closing = False

    async def _run_loop(self) -> None:
        """Maintain connections and relay messages."""
        logger.info("[Central Relay] _run_loop started")

        try:
            # Small yield to allow event loop to process other events
            await asyncio.sleep(0)
            logger.info("[Central Relay] Starting connection attempts")

            while self._running:
                had_exception = False
                try:
                    await self._connect_and_relay()
                except asyncio.CancelledError:
                    # CancelledError at this layer has two possible sources:
                    #
                    # 1. `stop()` flipped `self._running` to False. This is the
                    #    shutdown path and we must re-raise so the thread exits.
                    # 2. An in-flight `_close_connections()` (triggered by
                    #    `force_reconnect` / `update_token`) cancelled a task
                    #    that aiohttp was holding the cancellation for - e.g.
                    #    a `session.get(...)` wedged inside `_resolve_host` on
                    #    a flaky network. aiohttp propagates that cancellation
                    #    up through `_handle_central_sse`, which lands here.
                    #    In that case we very much want to stay in the loop and
                    #    reconnect - killing the thread here means every
                    #    subsequent `/refresh-relay` POST is a no-op because
                    #    there's no loop left to service the token_updated
                    #    event.
                    #
                    # `self._running` is our authoritative signal for (1). If
                    # it's still True, treat the cancellation as a reconnect
                    # request and loop around.
                    if not self._running:
                        logger.info(
                            "[Central Relay] _run_loop cancelled (stop requested)"
                        )
                        raise
                    logger.info(
                        "[Central Relay] Connect attempt cancelled mid-flight "
                        "(likely from force_reconnect / token update); restarting loop"
                    )
                    had_exception = True
                    self._set_state(
                        RelayState.RECONNECTING,
                        "Restarting after cancelled connect",
                    )
                except Exception as e:
                    logger.warning(
                        f"[Central Relay] Connection attempt failed with exception: {type(e).__name__}: {e}"
                    )
                    had_exception = True
                    self._connection_attempts += 1
                    if self._connection_attempts <= 3:
                        self._set_state(
                            RelayState.RECONNECTING, f"Connection failed: {e}"
                        )
                    else:
                        self._set_state(
                            RelayState.ERROR,
                            f"Connection failed after {self._connection_attempts} attempts: {e}",
                        )

                if self._running and had_exception:
                    # Only wait after connection failures, not after normal returns
                    # (e.g., when token update triggered reconnection)
                    self._token_updated.clear()
                    try:
                        await asyncio.wait_for(
                            self._token_updated.wait(), timeout=RECONNECT_INTERVAL
                        )
                    except asyncio.TimeoutError:
                        pass

        except asyncio.CancelledError:
            logger.info("[Central Relay] Run loop cancelled (stop requested)")
            raise
        except Exception as e:
            logger.error(f"[Central Relay] Unexpected error in run loop: {e}")
            raise

    async def _connect_and_relay(self) -> None:
        """Connect to both servers and relay messages."""
        logger.info("[Central Relay] _connect_and_relay() starting")

        # Always refresh the token on each connection attempt
        self._refresh_token()

        if not self.hf_token:
            self._set_state(
                RelayState.WAITING_FOR_TOKEN,
                "Login to HuggingFace to enable remote access",
            )
            # Wait longer when no token - user needs to log in
            self._token_updated.clear()
            try:
                await asyncio.wait_for(
                    self._token_updated.wait(), timeout=TOKEN_CHECK_INTERVAL
                )
                logger.info(
                    "[Central Relay] Token update received while waiting, will attempt connection"
                )
            except asyncio.TimeoutError:
                logger.debug("[Central Relay] Token check timeout, will re-check")
            return

        # Create HTTP session for central server
        self._http_session = aiohttp.ClientSession()

        # Connect to local GStreamer signaling (WebSocket) with timeout
        self._set_state(RelayState.CONNECTING, "Connecting to local WebRTC...")
        logger.info(
            f"[Central Relay] Attempting to connect to local websocket: {self.local_uri}"
        )
        try:
            self._local_ws = await asyncio.wait_for(
                websockets.connect(
                    self.local_uri,
                    ping_interval=None,
                    ping_timeout=None,
                ),
                timeout=LOCAL_WS_CONNECT_TIMEOUT,
            )
            logger.info("[Central Relay] Local websocket connection established")
        except asyncio.TimeoutError:
            logger.error(
                f"[Central Relay] Local WebRTC connection timeout after {LOCAL_WS_CONNECT_TIMEOUT}s"
            )
            self._set_state(
                RelayState.ERROR,
                f"Local WebRTC connection timeout after {LOCAL_WS_CONNECT_TIMEOUT}s",
            )
            await self._http_session.close()
            self._http_session = None
            raise
        except Exception as e:
            logger.error(f"[Central Relay] Local WebRTC connection failed: {e}")
            self._set_state(RelayState.ERROR, f"Local WebRTC unavailable: {e}")
            await self._http_session.close()
            self._http_session = None
            raise

        # Wait for welcome message from local websocket to verify connection is working
        self._local_welcome_received = asyncio.Event()
        logger.info(
            "[Central Relay] Waiting for welcome message from local websocket..."
        )
        try:
            # Start reading local messages in background to receive welcome
            local_task = asyncio.create_task(self._handle_local_messages())
            logger.info("[Central Relay] Local message handler task started")

            # Wait for welcome with timeout
            try:
                await asyncio.wait_for(
                    self._local_welcome_received.wait(),
                    timeout=LOCAL_WS_WELCOME_TIMEOUT,
                )
                logger.info(
                    "[Central Relay] Local WebRTC connection verified (welcome received)"
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"[Central Relay] Welcome message timeout after {LOCAL_WS_WELCOME_TIMEOUT}s"
                )
                local_task.cancel()
                try:
                    await local_task
                except asyncio.CancelledError:
                    pass
                self._set_state(
                    RelayState.ERROR,
                    f"Local WebRTC did not respond within {LOCAL_WS_WELCOME_TIMEOUT}s",
                )
                raise

            # Now connect to central server and run all handlers.
            # Use wait with FIRST_COMPLETED so we can reconnect if any
            # handler exits. The mapping documents one-to-one why each
            # task is part of the race; adding a new reconnect trigger
            # is a one-line change.
            self._set_state(RelayState.CONNECTING, "Connecting to central server...")
            relay_tasks: dict[asyncio.Task[None], str] = {
                asyncio.create_task(
                    self._handle_central_sse()
                ): "Central SSE handler exited, will reconnect",
                local_task: "Local WebSocket handler exited, will reconnect",
                asyncio.create_task(
                    self._watch_for_token_update()
                ): "Token update triggered reconnect",
                asyncio.create_task(
                    self._producer_health_loop()
                ): "Producer health check triggered reconnect",
                asyncio.create_task(
                    self._heartbeat_loop()
                ): "Heartbeat loop exited unexpectedly, will reconnect",
            }

            try:
                done, pending = await asyncio.wait(
                    relay_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    logger.info("[Central Relay] %s", relay_tasks[task])
                    if task.exception():
                        logger.warning(
                            "[Central Relay] Task %s raised: %s",
                            task.get_name(),
                            task.exception(),
                        )

                # Cancel remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Update state to show we're reconnecting (unless already set)
                if self._state == RelayState.CONNECTED:
                    self._set_state(
                        RelayState.RECONNECTING, "Connection lost, reconnecting..."
                    )
            except asyncio.CancelledError:
                # Cancel all tasks if we're cancelled
                for task in relay_tasks:
                    task.cancel()
                raise
        finally:
            await self._close_connections()

    async def _watch_for_token_update(self) -> None:
        """Watch for token updates and close connections to trigger reconnect."""
        await self._token_updated.wait()
        logger.debug(
            "[Central Relay] Token update signal received, closing connections"
        )
        await self._close_connections()

    def _build_producer_meta(self) -> dict[str, Any]:
        """Assemble the ``meta`` dict advertised to central listeners.

        Single source of truth for **what** we tell the world about
        ourselves. Kept separate from ``_producer_status_payload`` so
        future fields (``install_id``, ``capabilities``, ``health``,
        ``version``, ...) extend this method without disturbing the
        envelope shape; tests target this function directly.

        Today's surface is intentionally minimal:

          * ``name``: user-facing label (matches the central's
            ``robotName`` field on ``/api/robot-status``).
          * ``transport``: ``"usb"`` | ``"wifi"`` so the mobile picker
            can badge a listing without introspecting via a separate
            channel.
          * ``hardware_id`` (when available): SHA-256 prefix of the
            Pollen audio device's USB serial - stable per physical
            robot across OS reinstalls and renames. Lets the mobile
            picker dedupe a central listing against the same robot's
            BLE / loopback row, and serves as a short visible "robot
            tag" the user can recognise across sessions (the bare
            ``peerId`` rotates on every relay reconnect, so it is a
            poor display key). Omitted when the daemon runs without a
            Reachy attached (``get_hardware_id()`` returns ``None``);
            consumers must treat ``undefined`` as "no stable id" and
            fall back to whatever they already had.
        """
        meta: dict[str, Any] = {
            "name": self.robot_name,
            "transport": self.transport,
        }
        hw = get_hardware_id()
        if hw is not None:
            meta["hardware_id"] = hw
        return meta

    def _producer_status_payload(self) -> dict[str, Any]:
        """Single source of truth for our ``setPeerStatus`` payload.

        Used both by the post-welcome registration round-trip and by
        the periodic heartbeat re-emission. Keeping them DRY ensures
        a future field (e.g. ``install_id`` for the central's
        last-writer-wins dedup, ``capabilities`` for app gating, ...)
        can never go out of sync between the two paths.
        """
        return {
            "type": "setPeerStatus",
            "roles": ["producer"],
            "meta": self._build_producer_meta(),
        }

    @staticmethod
    def _negotiate_heartbeat_interval(welcome_msg: dict[str, Any]) -> float:
        """Pick the cadence at which we re-emit setPeerStatus.

        Priority order:
          1. ``recommended_heartbeat_interval_seconds`` field from the
             welcome (canonical signal: central advertises whatever
             cadence its server-side ``LEASE_SECONDS`` env tunes to,
             typically ``LEASE / 3``).
          2. ``lease_seconds`` field from the welcome divided by 3
             (older centrals that expose lease but not the recommended
             cadence directly).
          3. ``HEARTBEAT_DEFAULT_INTERVAL`` for pre-negotiation
             centrals that expose neither.

        Rungs 1 and 2 are passed through ``_clamp_heartbeat_interval``
        so a misconfigured central can neither ask us to spam (say,
        0.1 s) nor lull us into a cadence so slow we'd be evicted
        before the next heartbeat fires (say, 600 s).
        """
        raw = welcome_msg.get("recommended_heartbeat_interval_seconds")
        if isinstance(raw, (int, float)) and raw > 0:
            return _clamp_heartbeat_interval(float(raw))
        lease = welcome_msg.get("lease_seconds")
        if isinstance(lease, (int, float)) and lease > 0:
            return _clamp_heartbeat_interval(float(lease) / 3.0)
        return HEARTBEAT_DEFAULT_INTERVAL

    async def _heartbeat_loop(self) -> None:
        """Re-emit setPeerStatus periodically to keep the central lease alive.

        Central runs a TTL sweeper: a producer that does not generate
        inbound traffic (POST /send) for more than ``LEASE_SECONDS`` is
        evicted from its in-memory tables, even if the SSE channel is
        perfectly healthy. Half-open sockets (Wi-Fi yanked, NAT
        rebinding, captive portal sleep) are the prime offenders -
        the local TCP stack absorbs server-pushed keepalives silently
        for minutes and the daemon never realises it's already a ghost.

        This loop is the daemon-side half of the contract:
          * Cadence is set by ``_negotiate_heartbeat_interval`` from
            the SSE welcome and stored on ``_heartbeat_interval_seconds``.
            We re-read it on every iteration so the negotiated value
            propagates as soon as the welcome lands, even though this
            loop is spawned in parallel with the SSE handler that
            processes welcome (the first iteration uses the default,
            every subsequent one uses the negotiated value).
          * The ``await sleep(interval)`` happens BEFORE each re-emit,
            so the first heartbeat fires one interval after the loop
            starts (the post-welcome registration just refreshed
            ``last_seen``; firing again immediately would be wasteful).
          * Each re-emit reuses ``_producer_status_payload`` so we
            stay byte-identical to the registration payload. Central
            is idempotent under repeated setPeerStatus from the same peer.
          * A failed heartbeat is logged at DEBUG, not ERROR: the
            split-brain detector (``_producer_health_loop``) is the
            authoritative recovery path. If central truly dropped us,
            the next robot-status poll will notice and trigger a
            ``force_reconnect()``; redundant teardown from this loop
            would only fight that mechanism.
          * Exits on cancellation when ``_connect_and_relay``'s
            FIRST_COMPLETED race tears all sibling tasks down.
        """
        last_logged_interval: Optional[float] = None

        while self._running:
            interval = self._heartbeat_interval_seconds
            if interval != last_logged_interval:
                logger.info("[Central Relay] heartbeat loop interval=%.1fs", interval)
                last_logged_interval = interval

            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise

            if self._state != RelayState.CONNECTED:
                # Reconnecting / errored: nothing to refresh. Once we
                # land back in CONNECTED, this loop has already been
                # cancelled and respawned with a freshly negotiated
                # interval.
                continue

            try:
                await self._send_to_central(self._producer_status_payload())
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("[Central Relay] heartbeat re-emit failed: %s", e)

    async def _is_listed_as_producer(self) -> Optional[bool]:
        """Ask central whether we are still a registered producer.

        Returns:
            ``True`` if our ``_central_peer_id`` is in central's robot list
            for the authenticated user (healthy state).
            ``False`` if central acknowledged the request but we are not in
            the list (split-brain - the relay believes it is connected
            but the producer registration was lost on central's side).
            ``None`` if the check could not run (no token, no peer id, no
            HTTP session, or transient HTTP/network failure). Callers
            treat ``None`` as "don't update health counters" so a brief
            central hiccup never triggers a needless reconnect storm.

        """
        if not self._http_session or not self.hf_token or not self._central_peer_id:
            return None
        url = f"{self.central_uri}/api/robot-status"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        timeout = aiohttp.ClientTimeout(total=PRODUCER_HEALTH_CHECK_TIMEOUT)
        try:
            async with self._http_session.get(
                url, headers=headers, timeout=timeout
            ) as response:
                if response.status != 200:
                    logger.debug(
                        "[Central Relay] producer health check returned HTTP %s",
                        response.status,
                    )
                    return None
                data = await response.json()
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.debug("[Central Relay] producer health check transient error: %s", e)
            return None
        except Exception as e:
            logger.warning(
                "[Central Relay] producer health check unexpected error: %s", e
            )
            return None

        robots = data.get("robots") if isinstance(data, dict) else None
        if not isinstance(robots, list):
            return None
        peer_ids = {robot.get("peerId") for robot in robots if isinstance(robot, dict)}
        return self._central_peer_id in peer_ids

    async def _producer_health_loop(self) -> None:
        """Self-heal split-brain states by polling /api/robot-status.

        Without this, a relay whose ``setPeerStatus`` was cancelled
        mid-flight (token rotation, DNS hiccup, transient HTTP error)
        keeps an SSE channel open and reports state=CONNECTED even
        though central has no record of it as a producer. From the
        outside the robot looks online but no client can call it - the
        only recovery used to be SSH+systemctl or a manual POST to
        /api/hf-auth/refresh-relay.

        Loop semantics:
          * Wait ``PRODUCER_HEALTH_CHECK_INITIAL_DELAY`` after start so
            the post-welcome ``setPeerStatus`` round-trip has time to
            land in central's index.
          * Every ``PRODUCER_HEALTH_CHECK_INTERVAL`` seconds, ask
            central whether we are still listed.
          * Tolerate up to ``PRODUCER_HEALTH_CHECK_MAX_MISSES`` consecutive
            "not listed" answers before self-triggering a
            ``force_reconnect()``. This avoids reconnect storms on
            harmless central blips.
          * Exit on cancellation (``_connect_and_relay``'s
            FIRST_COMPLETED race tears us down on every reconnect).
        """
        try:
            await asyncio.sleep(PRODUCER_HEALTH_CHECK_INITIAL_DELAY)
        except asyncio.CancelledError:
            raise

        consecutive_misses = 0
        while self._running:
            try:
                await asyncio.sleep(PRODUCER_HEALTH_CHECK_INTERVAL)
            except asyncio.CancelledError:
                raise

            if self._state != RelayState.CONNECTED:
                # Don't probe while we're already reconnecting / in error -
                # there's nothing useful we can do about the listing state
                # until the SSE handshake settles again.
                consecutive_misses = 0
                continue

            listed = await self._is_listed_as_producer()
            if listed is True:
                consecutive_misses = 0
                continue
            if listed is None:
                # Transient error - keep our streak intact rather than
                # padding it with noise.
                continue

            consecutive_misses += 1
            logger.warning(
                "[Central Relay] producer not listed on central "
                "(miss %s/%s, peer_id=%s)",
                consecutive_misses,
                PRODUCER_HEALTH_CHECK_MAX_MISSES,
                self._central_peer_id,
            )
            if consecutive_misses >= PRODUCER_HEALTH_CHECK_MAX_MISSES:
                logger.warning(
                    "[Central Relay] split-brain detected, forcing reconnect"
                )
                # `force_reconnect` schedules a teardown via the thread
                # loop; this task gets cancelled by the FIRST_COMPLETED
                # race in `_connect_and_relay` and the next iteration
                # of `_run_loop` re-spawns us with a fresh peer_id.
                await self.force_reconnect()
                return

    async def _handle_central_sse(self) -> None:
        """Handle SSE events from central server."""
        if not self._http_session:
            return

        # Token goes in the Authorization header, never in the URL —
        # keeps it out of HF Space access logs and intermediate proxies.
        events_url = f"{self.central_uri}/events"
        headers = {"Authorization": f"Bearer {self.hf_token}"}

        try:
            # Use timeout for the initial connection
            timeout = aiohttp.ClientTimeout(
                total=None, connect=10, sock_read=SSE_READ_TIMEOUT
            )
            async with self._http_session.get(
                events_url, headers=headers, timeout=timeout
            ) as response:
                if response.status == 401:
                    self._set_state(
                        RelayState.ERROR, "Authentication failed - token may be invalid"
                    )
                    return
                elif response.status != 200:
                    self._set_state(
                        RelayState.ERROR,
                        f"Central server returned HTTP {response.status}",
                    )
                    return

                # Connection successful - will set CONNECTED after welcome message
                self._connection_attempts = 0

                # Read lines with timeout to detect dead connections
                while self._running:
                    try:
                        line = await asyncio.wait_for(
                            response.content.readline(), timeout=SSE_READ_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"[Central Relay] SSE read timeout after {SSE_READ_TIMEOUT}s - connection may be dead"
                        )
                        self._set_state(
                            RelayState.RECONNECTING,
                            "Connection timeout, reconnecting...",
                        )
                        return

                    if not line:
                        # Empty line means connection closed
                        logger.info("[Central Relay] SSE connection closed by server")
                        return

                    line_str = line.decode("utf-8").strip()

                    if line_str.startswith("data:"):
                        data = line_str[5:].strip()
                        if data:
                            try:
                                msg = json.loads(data)
                                await self._process_central_message(msg)
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"[Central Relay] Invalid JSON from central: {data[:100]}"
                                )

        except asyncio.CancelledError:
            raise
        except aiohttp.ClientError as e:
            self._set_state(RelayState.ERROR, f"Central server unreachable: {e}")
        except Exception as e:
            logger.error(f"[Central Relay] Error in central SSE: {e}")
        finally:
            # Clean up all sessions when central connection drops
            if self._local_to_central_session:
                logger.info(
                    f"[Central Relay] Central connection lost, cleaning up {len(self._local_to_central_session)} sessions"
                )
                for local_session_id in list(self._local_to_central_session.keys()):
                    await self._send_to_local(
                        {"type": "endSession", "sessionId": local_session_id}
                    )

    async def _handle_local_messages(self) -> None:
        """Handle messages from local GStreamer signaling."""
        if not self._local_ws:
            return

        try:
            async for message in self._local_ws:
                try:
                    message_str = (
                        message if isinstance(message, str) else message.decode("utf-8")
                    )
                    msg = json.loads(message_str)
                    await self._process_local_message(msg)
                except json.JSONDecodeError:
                    logger.warning(
                        f"[Central Relay] Invalid JSON from local GStreamer: {str(message)[:100]}"
                    )
        except websockets.ConnectionClosed:
            logger.info("[Central Relay] Local GStreamer WebSocket connection closed")
        except Exception as e:
            logger.error(
                f"[Central Relay] Error handling local GStreamer messages: {e}"
            )

    async def _send_to_central(self, msg: dict[str, Any]) -> None:
        """Send a message to the central server via HTTP POST."""
        msg_type = msg.get("type", "?")
        if not self._http_session or not self.hf_token:
            logger.warning(
                "[Central Relay] _send_to_central skipped (type=%s, http_session=%s, hf_token=%s)",
                msg_type,
                bool(self._http_session),
                bool(self.hf_token),
            )
            return

        # Token goes in the Authorization header only, not the URL.
        send_url = f"{self.central_uri}/send"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        try:
            async with self._http_session.post(
                send_url, json=msg, headers=headers
            ) as response:
                if response.status != 200:
                    body = ""
                    try:
                        body = (await response.text())[:300]
                    except Exception:
                        pass
                    logger.warning(
                        "[Central Relay] _send_to_central FAILED type=%s HTTP %s body=%r",
                        msg_type,
                        response.status,
                        body,
                    )
                else:
                    logger.info("[Central Relay] _send_to_central OK type=%s", msg_type)
        except Exception as e:
            logger.error(
                "[Central Relay] _send_to_central exception type=%s err=%s",
                msg_type,
                e,
            )

    async def _send_to_local(self, msg: dict[str, Any]) -> None:
        """Send a message to local GStreamer signaling."""
        if self._local_ws:
            try:
                await self._local_ws.send(json.dumps(msg))
            except Exception as e:
                logger.error(
                    f"[Central Relay] Failed to send message to local GStreamer: {e}"
                )

    async def _process_central_message(self, msg: dict[str, Any]) -> None:
        """Process a message from the central server."""
        msg_type = msg.get("type", "")
        logger.debug(f"[Central Relay] Received from central server: type={msg_type}")

        if msg_type == "welcome":
            # Received our peer ID from central server
            self._central_peer_id = msg.get("peerId")
            self._heartbeat_interval_seconds = self._negotiate_heartbeat_interval(msg)
            logger.info(
                "[Central Relay] central welcome received peer_id=%s; "
                "registering as producer name=%r, heartbeat=%.1fs",
                self._central_peer_id,
                self.robot_name,
                self._heartbeat_interval_seconds,
            )

            # Register as producer FIRST, then flip to CONNECTED. If we
            # set CONNECTED before producer registration, observers (UI,
            # mobile app, /relay-status pollers) can see "connected" while
            # central does not yet know we are a producer for this user,
            # which produces the desync described in /refresh-relay's
            # docstring.
            await self._send_to_central(self._producer_status_payload())

            self._set_state(
                RelayState.CONNECTED, f"Remote access enabled as '{self.robot_name}'"
            )

        elif msg_type == "list":
            # Ignore list messages - we're a producer
            pass

        elif msg_type == "startSession":
            # A client wants to connect - forward to local GStreamer
            client_peer_id: str = msg.get("peerId", "")
            session_id: Optional[str] = msg.get("sessionId")
            logger.info(
                f"[Central Relay] Received session request from remote client peer_id={client_peer_id} session_id={session_id}"
            )

            # Safety net: the central server is supposed to gate concurrent sessions,
            # but if one slips through (e.g. older central without the gate), enforce
            # single-session-at-a-time here so we never run two clients against one robot.
            if self._central_to_local_session or self._pending_central_sessions:
                logger.warning(
                    f"[Central Relay] Rejecting session {session_id}: a session is already active/pending"
                )
                if session_id:
                    await self._send_to_central(
                        {
                            "type": "endSession",
                            "sessionId": session_id,
                            "reason": "robot_busy_local",
                        }
                    )
                return

            # Gate on the robot lock: if a local Python app is running, the
            # lock will refuse our acquire. We also acquire proactively here
            # so a concurrent local-app start can't sneak in between the
            # check and the session handoff to local GStreamer.
            if self._robot_app_lock is not None:
                # holder_name is generic because central already tracks the
                # real consumer app name (via setPeerStatus meta) for its
                # own rejection messages; the daemon-side lock just needs
                # to know that *something* remote holds it.
                if not self._robot_app_lock.try_acquire_remote("remote"):
                    logger.warning(
                        f"[Central Relay] Rejecting session {session_id}: robot lock is held locally"
                    )
                    if session_id:
                        await self._send_to_central(
                            {
                                "type": "endSession",
                                "sessionId": session_id,
                                "reason": "robot_busy_local_app",
                            }
                        )
                    return

            # Store session mapping
            if session_id:
                # Check if we already have this session (duplicate request)
                if session_id in self._session_to_local_peer:
                    logger.warning(
                        f"[Central Relay] Duplicate session request for session_id={session_id}, ignoring"
                    )
                    return

                self._session_to_local_peer[session_id] = client_peer_id
                self._pending_central_sessions.append(session_id)
                logger.info(
                    f"[Central Relay] Pending sessions: {len(self._pending_central_sessions)}, tracked sessions: {len(self._session_to_local_peer)}"
                )

            # Request list of local producers to start session
            await self._send_to_local({"type": "list"})

        elif msg_type == "peer":
            # SDP/ICE from client - relay to local GStreamer
            central_session_id = msg.get("sessionId")
            if central_session_id and self._local_ws:
                # Translate central session ID to local session ID
                local_session_id = self._central_to_local_session.get(
                    central_session_id
                )
                if not local_session_id:
                    logger.warning(
                        f"[Central Relay] No local session mapping found for central_session_id={central_session_id}"
                    )
                    return

                local_msg = {
                    "type": "peer",
                    "sessionId": local_session_id,
                }
                if "sdp" in msg:
                    local_msg["sdp"] = msg["sdp"]
                if "ice" in msg:
                    local_msg["ice"] = msg["ice"]

                logger.debug(
                    f"[Central Relay] Relaying peer message: central_session={central_session_id} -> local_session={local_session_id}"
                )
                await self._send_to_local(local_msg)

        elif msg_type == "endSession":
            central_session_id = msg.get("sessionId")
            if central_session_id:
                logger.info(
                    f"[Central Relay] Session ended from central: central_session_id={central_session_id}"
                )
                self._session_to_local_peer.pop(central_session_id, None)
                # Also remove from pending if it never got started
                if central_session_id in self._pending_central_sessions:
                    self._pending_central_sessions.remove(central_session_id)
                # Translate and forward to local
                local_session_id = self._central_to_local_session.pop(
                    central_session_id, None
                )
                if local_session_id:
                    self._local_to_central_session.pop(local_session_id, None)
                    logger.info(
                        f"[Central Relay] Forwarding endSession to local: local_session_id={local_session_id}"
                    )
                    await self._send_to_local(
                        {"type": "endSession", "sessionId": local_session_id}
                    )
                logger.info(
                    f"[Central Relay] After cleanup - pending: {len(self._pending_central_sessions)}, active: {len(self._central_to_local_session)}"
                )
                # If no sessions remain, release the robot lock.
                if (
                    self._robot_app_lock is not None
                    and not self._central_to_local_session
                    and not self._pending_central_sessions
                ):
                    self._robot_app_lock.release_remote()

        elif msg_type == "peerStatusChanged":
            # Another peer changed status - ignore for producers
            pass

    async def _process_local_message(self, msg: dict[str, Any]) -> None:
        """Process a message from local GStreamer signaling."""
        msg_type = msg.get("type", "")
        logger.debug(f"[Central Relay] Received from local GStreamer: type={msg_type}")

        if msg_type == "welcome":
            # Received our peer ID from local GStreamer
            self._local_peer_id = msg.get("peerId")
            logger.info(
                f"[Central Relay] Connected to local GStreamer signaling server peer_id={self._local_peer_id}"
            )

            # Signal that local connection is verified
            if hasattr(self, "_local_welcome_received"):
                self._local_welcome_received.set()

            # Register as listener to receive producer announcements
            await self._send_to_local(
                {
                    "type": "setPeerStatus",
                    "roles": ["listener"],
                    "meta": {"name": "central-relay"},
                }
            )

        elif msg_type == "list":
            # List of local producers
            producers = msg.get("producers", [])
            if producers:
                self._local_producer_id = producers[0].get("id")
                logger.debug(
                    f"[Central Relay] Local GStreamer producer found: producer_id={self._local_producer_id}"
                )

                # Only start sessions for PENDING requests, not all tracked sessions
                for central_session_id in list(self._pending_central_sessions):
                    logger.info(
                        f"[Central Relay] Starting local session for pending central_session={central_session_id}"
                    )
                    await self._send_to_local(
                        {
                            "type": "startSession",
                            "peerId": self._local_producer_id,
                        }
                    )

        elif msg_type == "peerStatusChanged":
            peer_id = msg.get("peerId")
            roles = msg.get("roles", [])
            if "producer" in roles:
                self._local_producer_id = peer_id
                logger.debug(
                    f"[Central Relay] Local GStreamer producer registered: producer_id={peer_id}"
                )

        elif msg_type == "sessionStarted":
            local_session_id: Optional[str] = msg.get("sessionId")
            logger.info(
                f"[Central Relay] Local GStreamer session started: local_session_id={local_session_id}"
            )

            # Map local session ID to the pending central session ID
            if self._pending_central_sessions and local_session_id:
                central_session_id = self._pending_central_sessions.pop(0)
                self._local_to_central_session[local_session_id] = central_session_id
                self._central_to_local_session[central_session_id] = local_session_id
                logger.info(
                    f"[Central Relay] Session mapping established: local_session={local_session_id} <-> central_session={central_session_id}"
                )

        elif msg_type == "peer":
            # SDP/ICE from local GStreamer - relay to central
            local_session_id_peer: Optional[str] = msg.get("sessionId")
            if local_session_id_peer:
                # Translate local session ID to central session ID
                central_session_id_peer: Optional[str] = (
                    self._local_to_central_session.get(local_session_id_peer)
                )
                if not central_session_id_peer:
                    logger.warning(
                        f"[Central Relay] No central session mapping found for local_session_id={local_session_id_peer}"
                    )
                    return

                # Build message with translated session ID
                central_msg: dict[str, Any] = {
                    "type": "peer",
                    "sessionId": central_session_id_peer,
                }
                if "sdp" in msg:
                    central_msg["sdp"] = msg["sdp"]
                if "ice" in msg:
                    central_msg["ice"] = msg["ice"]

                logger.debug(
                    f"[Central Relay] Relaying peer message: local_session={local_session_id_peer} -> central_session={central_session_id_peer}"
                )
                await self._send_to_central(central_msg)

        elif msg_type == "endSession":
            local_session_id_end: Optional[str] = msg.get("sessionId")
            if local_session_id_end:
                logger.info(
                    f"[Central Relay] Session ended from local: local_session_id={local_session_id_end}"
                )
                # Translate and forward to central
                central_session_id_end: Optional[str] = (
                    self._local_to_central_session.pop(local_session_id_end, None)
                )
                if central_session_id_end:
                    self._central_to_local_session.pop(central_session_id_end, None)
                    self._session_to_local_peer.pop(central_session_id_end, None)
                    logger.info(
                        f"[Central Relay] Forwarding endSession to central: central_session_id={central_session_id_end}"
                    )
                    await self._send_to_central(
                        {"type": "endSession", "sessionId": central_session_id_end}
                    )
                logger.info(
                    f"[Central Relay] After cleanup - pending: {len(self._pending_central_sessions)}, active: {len(self._central_to_local_session)}"
                )
                # If no sessions remain, release the robot lock.
                if (
                    self._robot_app_lock is not None
                    and not self._central_to_local_session
                    and not self._pending_central_sessions
                ):
                    self._robot_app_lock.release_remote()


# Singleton instance for integration
_relay_instance: Optional[CentralSignalingRelay] = None


def get_relay() -> Optional[CentralSignalingRelay]:
    """Get the global relay instance.

    Returns:
        The relay instance, or None if not started

    """
    return _relay_instance


def get_relay_status() -> dict[str, Any]:
    """Get the current status of the central relay.

    Returns:
        A dict with state, message, and is_connected fields

    """
    if _relay_instance is None:
        return {
            "state": RelayState.STOPPED.value,
            "message": "Relay not initialized",
            "is_connected": False,
        }

    return {
        "state": _relay_instance.state.value,
        "message": _relay_instance.state_message,
        "is_connected": _relay_instance.state == RelayState.CONNECTED,
    }


async def start_central_relay(
    hf_token: Optional[str] = None,
    robot_name: str = "reachymini",
    transport: str = "wifi",
    central_uri: str = CENTRAL_SIGNALING_SERVER,
    on_state_change: Optional[Callable[[RelayState, Optional[str]], None]] = None,
    robot_app_lock: Optional[RobotAppLock] = None,
) -> CentralSignalingRelay:
    """Start the central signaling relay.

    Args:
        hf_token: HuggingFace token for authentication (will auto-refresh)
        robot_name: Name to register as producer
        transport: Producer ``meta.transport`` (``"usb"`` | ``"wifi"``).
            See ``CentralSignalingRelay.__init__`` for semantics.
        central_uri: Central server URI
        on_state_change: Callback when connection state changes
        robot_app_lock: Shared lock coordinating local vs remote robot access.

    Returns:
        The relay instance

    """
    global _relay_instance

    if _relay_instance is not None:
        return _relay_instance

    # Try to get HF token if not provided
    if hf_token is None:
        try:
            from huggingface_hub import get_token

            hf_token = get_token()
        except Exception:
            pass

    _relay_instance = CentralSignalingRelay(
        central_uri=central_uri,
        hf_token=hf_token,
        robot_name=robot_name,
        transport=transport,
        on_state_change=on_state_change,
        robot_app_lock=robot_app_lock,
    )
    await _relay_instance.start()
    return _relay_instance


async def stop_central_relay() -> None:
    """Stop the central signaling relay."""
    global _relay_instance

    if _relay_instance:
        await _relay_instance.stop()
        _relay_instance = None


async def notify_token_change(new_token: Optional[str] = None) -> None:
    """Notify the relay of a token change (login/logout).

    This should be called from the HF auth endpoints when the user
    logs in or out. If new_token is None, it will be fetched from
    huggingface_hub.

    Args:
        new_token: The new token, or None to fetch from huggingface_hub

    """
    if _relay_instance is None:
        logger.debug("[Central Relay] No relay instance, ignoring token change")
        return

    if new_token is None:
        try:
            from huggingface_hub import get_token

            new_token = get_token()
        except Exception:
            pass

    await _relay_instance.update_token(new_token)


async def notify_force_reconnect() -> bool:
    """Force the central signaling relay to drop and reconnect right now.

    Drops the relay's current connection and re-registers with the
    currently stored HF token. Recovery handle for the "zombie relay"
    state where ``/relay-status`` reports ``connected`` but
    ``/central-robot-status`` returns ``robots: []`` — central no
    longer lists this robot as a producer for the authenticated user
    while the relay still holds its SSE channel open. From the outside
    this manifests as "the robot is online but no one can call it"
    until someone restarts the daemon.

    Common triggers:
      - the relay attached with a token that has since been rotated;
      - a transient error during ``setPeerStatus`` went unnoticed.

    Unlike ``notify_token_change``, this skips the
    ``old_token == new_token`` early-return and always reconnects,
    which is exactly what callers of ``POST /api/hf-auth/refresh-relay``
    need (their whole reason for invoking is the token did not change
    but the producer registration was lost). Goes through the relay's
    own reconnect path — works with any token shape currently stored
    (raw user tokens, OAuth access tokens) without re-validation.

    Returns:
        ``True`` if a reconnect was actually kicked off, ``False`` if
        there was no relay instance to reconnect (e.g. Lite-only build,
        daemon not yet up). HTTP callers (mobile auto-heal flow) must
        surface this distinction so they can stop waiting on a
        reconnect that will never happen.

    """
    if _relay_instance is None:
        logger.debug("[Central Relay] No relay instance, ignoring force reconnect")
        return False

    await _relay_instance.force_reconnect()
    return True
