"""Backend.wake_up stand-down on an already-awake robot.

Clients routinely send ``wake_up`` on session start. Since the idle-reset
handoff grace keeps the robot awake between sessions, that boot-time wake
would replay the emote (goto + "toudoum" + roll) on a robot that never
slept. ``wake_up`` must stand down when the motors are enabled and the
controller is actively holding the init pose - and still run in full from
any other state. The predicate reads the commanded targets, not the
measured pose, so per-unit calibration residual can't flip the answer
(hardware-verified in #1311 review).

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
    head_pose: np.ndarray | None,
    antennas: np.ndarray | None = None,
) -> SimpleNamespace:
    if antennas is None and head_pose is not None:
        antennas = Backend.INIT_ANTENNAS_JOINT_POSITIONS.copy()
    fake = SimpleNamespace(
        INIT_HEAD_POSE=Backend.INIT_HEAD_POSE,
        INIT_ANTENNAS_JOINT_POSITIONS=Backend.INIT_ANTENNAS_JOINT_POSITIONS,
        INIT_TARGET_ATOL=Backend.INIT_TARGET_ATOL,
        # The predicate reads the commanded targets; the full wake path still
        # measures the present pose to size its goto duration.
        target_head_pose=head_pose,
        target_antenna_joint_positions=antennas,
        get_current_head_pose=lambda: (
            head_pose if head_pose is not None else Backend.SLEEP_HEAD_POSE.copy()
        ),
        get_motor_control_mode=lambda: motor_mode,
        goto_target=AsyncMock(),
        play_sound=MagicMock(),
        _on_wake_up_callback=MagicMock(),
    )
    # Bind the real predicate so these tests exercise it, not a stub.
    fake.is_awake_at_init_pose = lambda: Backend.is_awake_at_init_pose(fake)
    return fake


def _far_pose() -> np.ndarray:
    """Build a pose 50 mm away from init - well past the target atol."""
    pose = Backend.INIT_HEAD_POSE.copy()
    pose[0, 3] += 0.05  # +50 mm along x
    return pose


@pytest.mark.asyncio
async def test_wake_up_stands_down_when_already_awake_at_init() -> None:
    """Enabled motors + init is the commanded target: no motion, no sound."""
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
async def test_wake_up_stands_down_despite_calibration_residual() -> None:
    """A goto's interpolation tail (< atol) must not defeat the stand-down.

    The predicate compares commanded targets exactly so hardware residual is
    out of the picture, but the target itself lands one eased tick short of
    the endpoint (play_move never evaluates at t=duration). That tail must
    stay inside INIT_TARGET_ATOL.
    """
    near_init = Backend.INIT_HEAD_POSE.copy()
    near_init[0, 3] += 0.005  # half the atol
    fake = _fake_backend(
        motor_mode=MotorControlMode.Enabled,
        head_pose=near_init,
    )
    await Backend.wake_up(fake)
    fake.goto_target.assert_not_awaited()
    fake.play_sound.assert_not_called()
    fake._on_wake_up_callback.assert_called_once()


@pytest.mark.asyncio
async def test_wake_up_runs_when_no_target_commanded_yet() -> None:
    """Enabled motors but no target ever commanded (fresh boot): full wake.

    Also covers --sim, where the mujoco backend hardcodes motor mode to
    Enabled: the None guard keeps wake_up_on_start audible there.
    """
    fake = _fake_backend(
        motor_mode=MotorControlMode.Enabled,
        head_pose=None,
    )
    await Backend.wake_up(fake)
    assert fake.goto_target.await_count >= 1
    fake.play_sound.assert_called_once_with("wake_up.wav")
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
    """Disabled motors + init target still in place: must wake in full.

    Locks the `and` in the stand-down condition: a limp robot whose last
    commanded target was init is asleep, not awake.
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
async def test_wake_up_force_replays_even_when_awake_at_init() -> None:
    """force=True bypasses the stand-down: full emote on an awake robot.

    Used by callers that enable the motors themselves right before waking
    (antenna touch on a sleeping robot): enabling first would otherwise
    satisfy the stand-down and swallow the wake they just decided on.
    """
    fake = _fake_backend(
        motor_mode=MotorControlMode.Enabled,
        head_pose=Backend.INIT_HEAD_POSE.copy(),
    )
    await Backend.wake_up(fake, force=True)
    assert fake.goto_target.await_count >= 1
    fake.play_sound.assert_called_once_with("wake_up.wav")
    fake._on_wake_up_callback.assert_called_once()


@pytest.mark.asyncio
async def test_wake_up_runs_when_antennas_off_init() -> None:
    """Enabled + head target at init but antennas targeted askew: must run.

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
