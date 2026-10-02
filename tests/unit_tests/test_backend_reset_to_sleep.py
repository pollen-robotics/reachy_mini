"""Backend.reset_to_sleep: sleep the robot from any inherited motor state.

``goto_sleep`` assumes position control. When the last app broke that
assumption - gravity compensation, global torque cut, or a per-motor
``set_torque(ids=...)`` - it degrades silently: the control loop ignores the
position targets, so the trajectory is written into the void and the closing
torque cut drops the head from wherever it was. ``reset_to_sleep`` re-torques
first, lifts to init, and only then sleeps.

Same lightweight-fake approach as ``test_backend_idle_reset``: the method only
touches a handful of ``self`` attributes, so it is exercised unbound.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from reachy_mini.daemon.backend.abstract import Backend
from reachy_mini.io.protocol import MotorControlMode


def _fake_backend(*, head_pose: np.ndarray | None = None) -> SimpleNamespace:
    """Build a backend whose motor + motion primitives record their call order."""
    calls: list[str] = []
    if head_pose is None:
        head_pose = Backend.SLEEP_HEAD_POSE.copy()

    fake = SimpleNamespace(
        INIT_HEAD_POSE=Backend.INIT_HEAD_POSE,
        INIT_ANTENNAS_JOINT_POSITIONS=Backend.INIT_ANTENNAS_JOINT_POSITIONS,
        SLEEP_HEAD_POSE=Backend.SLEEP_HEAD_POSE,
        SLEEP_POSE_MAGIC_ATOL=Backend.SLEEP_POSE_MAGIC_ATOL,
        current_head_pose=head_pose,
        get_current_head_pose=lambda: head_pose,
        calls=calls,
    )
    fake._quiesce_aim_sources = MagicMock(side_effect=lambda: calls.append("quiesce"))
    fake.set_motor_control_mode = MagicMock(
        side_effect=lambda mode: calls.append(f"mode:{mode.value}")
    )
    fake.goto_target = AsyncMock(side_effect=lambda *a, **kw: calls.append("goto"))
    fake.goto_sleep = AsyncMock(side_effect=lambda: calls.append("sleep"))
    return fake


def _drooped_pose() -> np.ndarray:
    """Build a head pose hanging well away from both init and sleep."""
    pose = Backend.INIT_HEAD_POSE.copy()
    pose[2, 3] -= 0.03
    pose[0, 3] += 0.02
    return pose


def _far_pose() -> np.ndarray:
    """Build a head pose far enough that an unclamped duration would overshoot."""
    pose = Backend.INIT_HEAD_POSE.copy()
    pose[0, 3] += 0.2
    return pose


@pytest.mark.asyncio
async def test_torque_is_restored_before_any_motion() -> None:
    """The ordering that makes the whole thing work, locked in.

    Enabling after the goto would mean commanding a limp robot; quiescing
    after the enable would let head tracking re-aim between the two.
    """
    fake = _fake_backend(head_pose=_drooped_pose())
    await Backend.reset_to_sleep(fake)
    assert fake.calls == [
        "quiesce",
        f"mode:{MotorControlMode.Enabled.value}",
        "goto",
        "sleep",
    ]


@pytest.mark.asyncio
async def test_lift_targets_the_init_pose() -> None:
    """The lift goes to init (head AND antennas), not straight to sleep.

    Collapsing every inherited pose into init is what makes the following
    sleep trajectory predictable, since that is the start it was tuned for.
    """
    fake = _fake_backend(head_pose=_drooped_pose())
    await Backend.reset_to_sleep(fake)

    args, kwargs = fake.goto_target.call_args
    assert np.allclose(args[0], Backend.INIT_HEAD_POSE)
    assert np.allclose(kwargs["antennas"], Backend.INIT_ANTENNAS_JOINT_POSITIONS)


@pytest.mark.asyncio
async def test_lift_duration_stays_within_bounds() -> None:
    """Never zero-length (silent no-op) and never a crawl, whatever the distance."""
    for pose in (Backend.INIT_HEAD_POSE.copy(), _drooped_pose(), _far_pose()):
        fake = _fake_backend(head_pose=pose)
        await Backend.reset_to_sleep(fake)
        duration = fake.goto_target.call_args.kwargs["duration"]
        assert 0.3 <= duration <= 1.5


@pytest.mark.asyncio
async def test_sleep_still_runs_when_already_near_init() -> None:
    """A robot already at init must still be slept, not just re-torqued."""
    fake = _fake_backend(head_pose=Backend.INIT_HEAD_POSE.copy())
    await Backend.reset_to_sleep(fake)
    fake.goto_sleep.assert_awaited_once()


# ---------------------------------------------------------------------------
# is_at_sleep_pose: the predicate the idle reset skips on
# ---------------------------------------------------------------------------


def _sleep_pose_backend(head_pose: np.ndarray | None) -> SimpleNamespace:
    return SimpleNamespace(
        SLEEP_HEAD_POSE=Backend.SLEEP_HEAD_POSE,
        SLEEP_POSE_MAGIC_ATOL=Backend.SLEEP_POSE_MAGIC_ATOL,
        current_head_pose=head_pose,
        get_current_head_pose=lambda: head_pose,
    )


def test_is_at_sleep_pose_true_at_the_sleep_pose() -> None:
    """The pose the sleep trajectory ends on must read as asleep."""
    assert Backend.is_at_sleep_pose(_sleep_pose_backend(Backend.SLEEP_HEAD_POSE.copy()))


def test_is_at_sleep_pose_false_at_init() -> None:
    """Init is the canonical "awake" pose - it must never read as asleep."""
    assert not Backend.is_at_sleep_pose(
        _sleep_pose_backend(Backend.INIT_HEAD_POSE.copy())
    )


def test_is_at_sleep_pose_false_before_first_kinematics_update() -> None:
    """No measured pose yet: claim nothing, so the caller does the work."""
    assert not Backend.is_at_sleep_pose(_sleep_pose_backend(None))
