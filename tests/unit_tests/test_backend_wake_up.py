"""Backend.wake_up stand-down on an already-awake robot.

Clients routinely send ``wake_up`` on session start. Since the idle-reset
handoff grace keeps the robot awake between sessions, that boot-time wake
would replay the emote (goto + "toudoum" + roll) on a robot that never
slept. ``wake_up`` must stand down when the motors are enabled and the head
is already at the init pose - and still run in full from any other state.

Same lightweight-fake approach as ``test_backend_idle_reset``: the method
only touches a handful of ``self`` attributes, so it is exercised unbound.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from reachy_mini.daemon.backend.abstract import Backend
from reachy_mini.io.protocol import MotorControlMode


def _fake_backend(
    *,
    motor_mode: MotorControlMode,
    head_pose: np.ndarray,
    antennas: np.ndarray | None = None,
) -> SimpleNamespace:
    if antennas is None:
        antennas = Backend.INIT_ANTENNAS_JOINT_POSITIONS.copy()
    return SimpleNamespace(
        INIT_HEAD_POSE=Backend.INIT_HEAD_POSE,
        INIT_ANTENNAS_JOINT_POSITIONS=Backend.INIT_ANTENNAS_JOINT_POSITIONS,
        get_current_head_pose=lambda: head_pose,
        get_present_antenna_joint_positions=lambda: antennas,
        get_motor_control_mode=lambda: motor_mode,
        goto_target=AsyncMock(),
        play_sound=MagicMock(),
        _on_wake_up_callback=MagicMock(),
    )


def _far_pose() -> np.ndarray:
    """Build a pose ~50 magic-mm away from init - well past the 10 threshold."""
    pose = Backend.INIT_HEAD_POSE.copy()
    pose[0, 3] += 0.05  # +50 mm along x
    return pose


@pytest.mark.asyncio
async def test_wake_up_stands_down_when_already_awake_at_init() -> None:
    """Enabled motors + head at init pose: no motion, no sound."""
    fake = _fake_backend(
        motor_mode=MotorControlMode.Enabled,
        head_pose=Backend.INIT_HEAD_POSE.copy(),
    )
    await Backend.wake_up(fake)
    fake.goto_target.assert_not_awaited()
    fake.play_sound.assert_not_called()
    # The "robot is awake" hook keeps firing so a configured startup app
    # still launches whatever path the wake took.
    fake._on_wake_up_callback.assert_called_once()


@pytest.mark.asyncio
async def test_wake_up_runs_in_full_when_asleep() -> None:
    """Disabled motors (fresh boot / post-sleep): the full emote plays."""
    fake = _fake_backend(
        motor_mode=MotorControlMode.Disabled,
        head_pose=Backend.SLEEP_HEAD_POSE.copy(),
    )
    await Backend.wake_up(fake)
    assert fake.goto_target.await_count >= 1
    fake.play_sound.assert_called_once_with("wake_up.wav")
    fake._on_wake_up_callback.assert_called_once()


@pytest.mark.asyncio
async def test_wake_up_runs_when_enabled_but_off_pose() -> None:
    """Enabled but parked far from init (e.g. crashed app): still wakes."""
    fake = _fake_backend(
        motor_mode=MotorControlMode.Enabled,
        head_pose=_far_pose(),
    )
    await Backend.wake_up(fake)
    assert fake.goto_target.await_count >= 1
    fake.play_sound.assert_called_once_with("wake_up.wav")


@pytest.mark.asyncio
async def test_wake_up_runs_when_disabled_even_at_init_pose() -> None:
    """Disabled motors + head at init: must still wake in full.

    Locks the `and` in the stand-down condition: a limp robot that happens
    to rest at the init pose is asleep, not awake.
    """
    fake = _fake_backend(
        motor_mode=MotorControlMode.Disabled,
        head_pose=Backend.INIT_HEAD_POSE.copy(),
    )
    await Backend.wake_up(fake)
    assert fake.goto_target.await_count >= 1
    fake.play_sound.assert_called_once_with("wake_up.wav")
    fake._on_wake_up_callback.assert_called_once()


@pytest.mark.asyncio
async def test_wake_up_runs_when_antennas_off_init() -> None:
    """Enabled + head at init but antennas askew: the wake must run.

    The wake's goto is what re-homes the antennas; standing down here
    would leave them wherever the previous app parked them.
    """
    fake = _fake_backend(
        motor_mode=MotorControlMode.Enabled,
        head_pose=Backend.INIT_HEAD_POSE.copy(),
        antennas=Backend.SLEEP_ANTENNAS_JOINT_POSITIONS.copy(),
    )
    await Backend.wake_up(fake)
    assert fake.goto_target.await_count >= 1
    fake.play_sound.assert_called_once_with("wake_up.wav")
    fake._on_wake_up_callback.assert_called_once()
