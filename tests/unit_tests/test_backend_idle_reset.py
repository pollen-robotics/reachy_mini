"""Backend idle-reset behaviour: debounce, post-grace skip, and the
finally-guard that must not orphan a freshly rescheduled task.

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
from reachy_mini.io.protocol import MotorControlMode


def _fake_backend(
    *,
    motor_mode: MotorControlMode = MotorControlMode.Enabled,
    shutting_down: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        IDLE_RESET_DEBOUNCE_S=0.02,
        is_shutting_down=shutting_down,
        get_motor_control_mode=lambda: motor_mode,
        goto_sleep=AsyncMock(),
        logger=SimpleNamespace(warning=lambda *a, **k: None),
        _idle_reset_task=None,
    )


@pytest.mark.asyncio
async def test_idle_reset_sleeps_after_debounce() -> None:
    """When the slot stays free past the grace period, goto_sleep runs."""
    fake = _fake_backend()
    task = asyncio.ensure_future(Backend._async_idle_reset(fake))
    fake._idle_reset_task = task
    await task
    fake.goto_sleep.assert_awaited_once()
    assert fake._idle_reset_task is None


@pytest.mark.asyncio
async def test_cancel_during_debounce_skips_motion() -> None:
    """A reconnect within the grace period cancels the reset before any motion."""
    fake = _fake_backend()
    task = asyncio.ensure_future(Backend._async_idle_reset(fake))
    fake._idle_reset_task = task
    await asyncio.sleep(0)  # let the task reach the debounce sleep
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    fake.goto_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_idle_reset_skips_when_already_disabled() -> None:
    """Re-check after the grace period: skip goto_sleep if already limp."""
    fake = _fake_backend(motor_mode=MotorControlMode.Disabled)
    task = asyncio.ensure_future(Backend._async_idle_reset(fake))
    fake._idle_reset_task = task
    await task
    fake.goto_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_idle_reset_skips_when_shutting_down() -> None:
    """Re-check after the grace period: skip goto_sleep if shutdown started."""
    fake = _fake_backend(shutting_down=True)
    task = asyncio.ensure_future(Backend._async_idle_reset(fake))
    fake._idle_reset_task = task
    await task
    fake.goto_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_finally_does_not_clobber_newer_task() -> None:
    """A cancelled task's finally must not orphan a freshly rescheduled one.

    Regression guard: without the ``is current_task()`` check, the old task's
    ``finally`` would blindly null ``_idle_reset_task``, so a later
    ``_cancel_idle_reset`` would miss the newer in-flight goto_sleep.
    """
    fake = _fake_backend()
    task_a = asyncio.ensure_future(Backend._async_idle_reset(fake))
    fake._idle_reset_task = task_a
    await asyncio.sleep(0)  # let task_a reach the debounce sleep

    # Simulate _cancel_idle_reset() + reschedule installing a newer handle.
    task_a.cancel()
    sentinel = object()
    fake._idle_reset_task = sentinel

    with pytest.raises(asyncio.CancelledError):
        await task_a

    assert fake._idle_reset_task is sentinel
