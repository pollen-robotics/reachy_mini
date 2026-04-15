"""Unit tests for :class:`reachy_mini.daemon.robot_lock.RobotLock`.

The lock coordinates two code paths that can grab the robot: local
Python apps launched by :class:`AppManager` (main asyncio loop) and
remote WebRTC clients routed through :class:`CentralSignalingRelay`
(its own thread + event loop). These tests exercise the state-machine
rules in isolation — no relay, no subprocess.

Integration with the relay is not covered here; it requires mocking
the relay's thread event loop and is tracked as tech debt.
"""

from __future__ import annotations

import asyncio

import pytest

from reachy_mini.daemon.robot_lock import RobotLock, RobotLockState


# ---------------------------------------------------------------------------
# Starting state
# ---------------------------------------------------------------------------


def test_starts_free() -> None:
    """A fresh lock is free and has no holder."""
    lock = RobotLock()
    status = lock.status()
    assert status.state == RobotLockState.FREE
    assert status.holder_name is None


# ---------------------------------------------------------------------------
# Local acquire / release
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_acquire_from_free() -> None:
    """Acquiring local from free transitions to local_app with the given name."""
    lock = RobotLock()
    await lock.acquire_local_evicting_remote("app_a")
    status = lock.status()
    assert status.state == RobotLockState.LOCAL_APP
    assert status.holder_name == "app_a"


@pytest.mark.asyncio
async def test_local_release_returns_to_free() -> None:
    """Releasing local returns to free and clears the holder name."""
    lock = RobotLock()
    await lock.acquire_local_evicting_remote("app_a")
    lock.release_local("app_a")
    status = lock.status()
    assert status.state == RobotLockState.FREE
    assert status.holder_name is None


@pytest.mark.asyncio
async def test_double_local_acquire_raises() -> None:
    """A second local acquire while a local app holds the lock must raise."""
    lock = RobotLock()
    await lock.acquire_local_evicting_remote("app_a")
    with pytest.raises(RuntimeError):
        await lock.acquire_local_evicting_remote("app_b")
    # State unchanged after the failed acquire.
    status = lock.status()
    assert status.state == RobotLockState.LOCAL_APP
    assert status.holder_name == "app_a"


def test_release_local_is_idempotent_when_free() -> None:
    """Releasing when the lock is already free is a silent no-op."""
    lock = RobotLock()
    lock.release_local("nonexistent")  # must not raise
    assert lock.status().state == RobotLockState.FREE


def test_release_local_is_no_op_when_remote_held() -> None:
    """Releasing local when a remote session holds the lock must not clear it."""
    lock = RobotLock()
    assert lock.try_acquire_remote("client1") is True
    lock.release_local("some_app")
    status = lock.status()
    assert status.state == RobotLockState.REMOTE_SESSION
    assert status.holder_name == "client1"


@pytest.mark.asyncio
async def test_release_local_tolerates_holder_mismatch() -> None:
    """If the release name differs from the current holder, the lock still releases.

    Protects against races after a rapid stop/start cycle where the
    subprocess monitor's ``finally`` might release under a different
    app name than the one that actually holds the lock.
    """
    lock = RobotLock()
    await lock.acquire_local_evicting_remote("app_a")
    lock.release_local("not_app_a")  # warning logged, still releases
    assert lock.status().state == RobotLockState.FREE


# ---------------------------------------------------------------------------
# Remote acquire / release
# ---------------------------------------------------------------------------


def test_remote_acquire_from_free() -> None:
    """Acquiring remote from free transitions to remote_session."""
    lock = RobotLock()
    assert lock.try_acquire_remote("relay") is True
    status = lock.status()
    assert status.state == RobotLockState.REMOTE_SESSION
    assert status.holder_name == "relay"


def test_remote_acquire_refused_while_remote_held() -> None:
    """A second remote acquire must fail; state is unchanged."""
    lock = RobotLock()
    assert lock.try_acquire_remote("client1") is True
    assert lock.try_acquire_remote("client2") is False
    status = lock.status()
    assert status.state == RobotLockState.REMOTE_SESSION
    assert status.holder_name == "client1"


@pytest.mark.asyncio
async def test_remote_acquire_refused_while_local_held() -> None:
    """A remote acquire must fail whenever a local app holds the lock."""
    lock = RobotLock()
    await lock.acquire_local_evicting_remote("app_a")
    assert lock.try_acquire_remote("client1") is False
    status = lock.status()
    assert status.state == RobotLockState.LOCAL_APP
    assert status.holder_name == "app_a"


def test_remote_release_returns_to_free() -> None:
    """Releasing remote returns to free."""
    lock = RobotLock()
    lock.try_acquire_remote("client1")
    lock.release_remote()
    assert lock.status().state == RobotLockState.FREE


def test_release_remote_is_idempotent_when_free() -> None:
    """Releasing when the lock is already free is a silent no-op."""
    lock = RobotLock()
    lock.release_remote()  # must not raise
    assert lock.status().state == RobotLockState.FREE


@pytest.mark.asyncio
async def test_release_remote_is_no_op_when_local_held() -> None:
    """Releasing remote when a local app holds the lock must not clear it."""
    lock = RobotLock()
    await lock.acquire_local_evicting_remote("app_a")
    lock.release_remote()
    status = lock.status()
    assert status.state == RobotLockState.LOCAL_APP
    assert status.holder_name == "app_a"


# ---------------------------------------------------------------------------
# Eviction: local acquire while a remote session holds the lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_acquire_evicts_remote_and_invokes_handler() -> None:
    """Local acquire atomically evicts a remote session and runs the handler."""
    lock = RobotLock()
    assert lock.try_acquire_remote("client1") is True

    calls: list[str] = []

    async def handler() -> None:
        # When the handler fires, the state must already be LOCAL_APP so
        # any incoming remote startSession is rejected. This is the
        # atomicity guarantee we want to verify.
        calls.append(f"state={lock.status().state.value}")

    lock.set_remote_eviction_handler(handler)
    await lock.acquire_local_evicting_remote("app_a")

    assert calls == ["state=local_app"]
    status = lock.status()
    assert status.state == RobotLockState.LOCAL_APP
    assert status.holder_name == "app_a"


@pytest.mark.asyncio
async def test_handler_not_invoked_when_acquiring_from_free() -> None:
    """No remote session to evict → handler must not run."""
    lock = RobotLock()

    called = asyncio.Event()

    async def handler() -> None:
        called.set()

    lock.set_remote_eviction_handler(handler)
    await lock.acquire_local_evicting_remote("app_a")

    # Give the event loop a turn in case the handler was scheduled.
    await asyncio.sleep(0)
    assert not called.is_set()


@pytest.mark.asyncio
async def test_handler_exception_is_swallowed_and_state_is_local() -> None:
    """If the eviction handler raises, the state still transitions to local_app.

    The handler is best-effort: once the lock is in ``local_app``, the
    relay will reject new remote sessions regardless, so a failed
    tear-down does not undo the acquire.
    """
    lock = RobotLock()
    assert lock.try_acquire_remote("client1") is True

    async def handler() -> None:
        raise RuntimeError("simulated tear-down failure")

    lock.set_remote_eviction_handler(handler)

    # Must not propagate the handler's exception.
    await lock.acquire_local_evicting_remote("app_a")
    status = lock.status()
    assert status.state == RobotLockState.LOCAL_APP
    assert status.holder_name == "app_a"


@pytest.mark.asyncio
async def test_clearing_handler_disables_eviction_callback() -> None:
    """Passing ``None`` to ``set_remote_eviction_handler`` clears it."""
    lock = RobotLock()
    assert lock.try_acquire_remote("client1") is True

    called = False

    async def handler() -> None:
        nonlocal called
        called = True

    lock.set_remote_eviction_handler(handler)
    lock.set_remote_eviction_handler(None)
    await lock.acquire_local_evicting_remote("app_a")
    assert called is False
