"""Backend idle-reset behaviour: debounce, skip, and the finally-guard.

The finally-guard must not orphan a freshly rescheduled task.
The idle-reset methods live on the abstract ``Backend`` and only touch a
handful of ``self`` attributes, so we exercise them as unbound methods against
a lightweight fake ``self`` (same approach as ``test_app_stop_sleep``) instead
of standing up a full backend.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from reachy_mini.daemon.backend.abstract import Backend
from reachy_mini.daemon.robot_app_lock import RobotAppLock
from reachy_mini.io.protocol import MotorControlMode

GRACE_S = 0.02


def _fake_backend(
    *,
    motor_mode: MotorControlMode = MotorControlMode.Enabled,
    at_sleep_pose: bool = False,
    shutting_down: bool = False,
) -> SimpleNamespace:
    fake = SimpleNamespace(
        IDLE_RESET_DEBOUNCE_S=GRACE_S,
        IDLE_RESET_HANDOFF_GRACE_S=GRACE_S * 10,
        is_shutting_down=shutting_down,
        get_motor_control_mode=lambda: motor_mode,
        is_at_sleep_pose=lambda: at_sleep_pose,
        reset_to_sleep=AsyncMock(),
        logger=SimpleNamespace(warning=lambda *a, **k: None),
        _idle_reset_task=None,
    )
    # Bind the real predicate so these tests exercise it, not a stub.
    fake._already_idle = lambda: Backend._already_idle(fake)
    return fake


@pytest.mark.asyncio
async def test_idle_reset_sleeps_after_debounce() -> None:
    """When the slot stays free past the grace period, the reset runs."""
    fake = _fake_backend()
    task = asyncio.ensure_future(Backend._async_idle_reset(fake, GRACE_S))
    fake._idle_reset_task = task
    await task
    fake.reset_to_sleep.assert_awaited_once()
    assert fake._idle_reset_task is None


@pytest.mark.asyncio
async def test_cancel_during_debounce_skips_motion() -> None:
    """A reconnect within the grace period cancels the reset before any motion."""
    fake = _fake_backend()
    task = asyncio.ensure_future(Backend._async_idle_reset(fake, GRACE_S))
    fake._idle_reset_task = task
    await asyncio.sleep(0)  # let the task reach the debounce sleep
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    fake.reset_to_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_idle_reset_skips_when_limp_at_the_sleep_pose() -> None:
    """Nothing to do: a clean leave sequence already left the robot down."""
    fake = _fake_backend(motor_mode=MotorControlMode.Disabled, at_sleep_pose=True)
    task = asyncio.ensure_future(Backend._async_idle_reset(fake, GRACE_S))
    fake._idle_reset_task = task
    await task
    fake.reset_to_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_idle_reset_runs_when_limp_away_from_the_sleep_pose() -> None:
    """The crashed-app signature: torque cut mid-pose, head left drooping.

    Regression guard for the motor-mode-only skip this replaced, which read
    "limp" as "already asleep" and left the robot hanging wherever it died.
    """
    fake = _fake_backend(motor_mode=MotorControlMode.Disabled, at_sleep_pose=False)
    task = asyncio.ensure_future(Backend._async_idle_reset(fake, GRACE_S))
    fake._idle_reset_task = task
    await task
    fake.reset_to_sleep.assert_awaited_once()


@pytest.mark.asyncio
async def test_idle_reset_runs_in_gravity_compensation() -> None:
    """Gravity compensation is awake, however limp it feels to the hand."""
    fake = _fake_backend(motor_mode=MotorControlMode.GravityCompensation)
    task = asyncio.ensure_future(Backend._async_idle_reset(fake, GRACE_S))
    fake._idle_reset_task = task
    await task
    fake.reset_to_sleep.assert_awaited_once()


@pytest.mark.asyncio
async def test_idle_reset_skips_when_shutting_down() -> None:
    """Re-check after the grace period: skip the reset if shutdown started."""
    fake = _fake_backend(shutting_down=True)
    task = asyncio.ensure_future(Backend._async_idle_reset(fake, GRACE_S))
    fake._idle_reset_task = task
    await task
    fake.reset_to_sleep.assert_not_awaited()


def test_handoff_release_gets_the_long_grace() -> None:
    """A hand-off release must schedule with the long grace, not the debounce.

    Regression guard for the mobile app dropping its session so an app iframe
    can take the slot: the short debounce expires long before a Space has
    finished cold-starting, and the robot would fall asleep in that gap.
    """
    scheduled: list[tuple] = []
    fake = _fake_backend()
    fake._maybe_start_idle_reset = lambda *args: None
    fake._log_loop = SimpleNamespace(
        call_soon_threadsafe=lambda fn, *args: scheduled.append((fn, args))
    )

    Backend.request_idle_reset(fake, expect_handoff=True)
    assert scheduled == [
        (fake._maybe_start_idle_reset, (fake.IDLE_RESET_HANDOFF_GRACE_S,))
    ]

    scheduled.clear()
    Backend.request_idle_reset(fake, expect_handoff=False)
    assert scheduled == [(fake._maybe_start_idle_reset, (fake.IDLE_RESET_DEBOUNCE_S,))]


def _wire_lock_to_backend(fake: SimpleNamespace) -> RobotAppLock:
    """Wire a real RobotAppLock to the fake backend the way the daemon does.

    Free transition -> ``request_idle_reset(expect_handoff=...)``; remote
    acquire -> ``cancel_idle_reset()``. The fake gets the real scheduling
    machinery bound so an actual asyncio task carries the grace period.
    """
    fake.ready = SimpleNamespace(is_set=lambda: True)
    fake._log_loop = asyncio.get_running_loop()
    fake._maybe_start_idle_reset = lambda grace_s: Backend._maybe_start_idle_reset(
        fake, grace_s
    )
    fake._cancel_idle_reset = lambda: Backend._cancel_idle_reset(fake)
    fake._async_idle_reset = lambda grace_s: Backend._async_idle_reset(fake, grace_s)

    lock = RobotAppLock()
    lock.set_on_became_free_handler(
        lambda expect_handoff: Backend.request_idle_reset(
            fake, expect_handoff=expect_handoff
        )
    )
    lock.set_on_remote_acquired_handler(lambda: Backend.cancel_idle_reset(fake))
    return lock


@pytest.mark.asyncio
async def test_remote_acquire_cancels_pending_handoff_grace_reset() -> None:
    """A successor taking the slot must cancel the pending handoff-grace reset.

    This is the daemon seam end to end: release-for-handoff schedules a real
    reset task with the long grace, and the successor's ``try_acquire_remote``
    - not its first data-channel command - cancels it before any motion.
    """
    fake = _fake_backend()
    lock = _wire_lock_to_backend(fake)

    assert lock.try_acquire_remote("predecessor") is True
    lock.release_remote(expect_handoff=True)
    await asyncio.sleep(0)  # run the posted _maybe_start_idle_reset
    assert fake._idle_reset_task is not None  # the grace timer is really armed

    assert lock.try_acquire_remote("successor") is True
    await asyncio.sleep(0)  # run the posted _cancel_idle_reset

    # Wait well past the handoff grace: the cancelled reset must never fire.
    await asyncio.sleep(fake.IDLE_RESET_HANDOFF_GRACE_S * 1.5)
    fake.reset_to_sleep.assert_not_awaited()
    assert fake._idle_reset_task is None


@pytest.mark.asyncio
async def test_handoff_grace_reset_fires_when_no_successor_arrives() -> None:
    """Positive control for the seam: no successor, the robot goes to sleep.

    Proves the wiring in the test above would have slept the robot - so the
    assert_not_awaited there is the acquire's doing, not a broken setup.
    """
    fake = _fake_backend()
    lock = _wire_lock_to_backend(fake)

    assert lock.try_acquire_remote("predecessor") is True
    lock.release_remote(expect_handoff=True)
    await asyncio.sleep(0)

    await asyncio.sleep(fake.IDLE_RESET_HANDOFF_GRACE_S * 1.5)
    fake.reset_to_sleep.assert_awaited_once()


@pytest.mark.asyncio
async def test_finally_does_not_clobber_newer_task() -> None:
    """A cancelled task's finally must not orphan a freshly rescheduled one.

    Regression guard: without the ``is current_task()`` check, the old task's
    ``finally`` would blindly null ``_idle_reset_task``, so a later
    ``_cancel_idle_reset`` would miss the newer in-flight reset.
    """
    fake = _fake_backend()
    task_a = asyncio.ensure_future(Backend._async_idle_reset(fake, GRACE_S))
    fake._idle_reset_task = task_a
    await asyncio.sleep(0)  # let task_a reach the debounce sleep

    # Simulate _cancel_idle_reset() + reschedule installing a newer handle.
    task_a.cancel()
    sentinel = object()
    fake._idle_reset_task = sentinel

    with pytest.raises(asyncio.CancelledError):
        await task_a

    assert fake._idle_reset_task is sentinel
