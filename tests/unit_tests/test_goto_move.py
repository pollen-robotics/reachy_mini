"""GotoMove antenna trajectories must respect the servo's single-turn space."""

import numpy as np

from reachy_mini.motion.goto import GotoMove
from reachy_mini.utils.interpolation import InterpolationTechnique


def _evaluate_antennas(move, duration, steps=500):
    trajectory = []
    for i in range(steps + 1):
        _, antennas, _ = move.evaluate(duration * i / steps)
        trajectory.append(antennas)
    return np.array(trajectory)


def test_goto_wraps_multi_turn_antenna_start():
    """A raw multi-turn start must not make the command cross +-180 deg.

    The antenna encoders accumulate whole turns while handled floppy; a
    goto whose start was captured from such a reading (physical +82 deg
    read as -278 deg) interpolates the COMMAND across the -180 deg wrap
    boundary, which the single-turn servo interprets as a full-turn
    teleport of the target: the antenna violently sweeps mid-goto (seen
    live 2026-07-06, ~1800 deg/s at 42% of a 5 s goto). Wrapping the
    start into one turn makes the straight path boundary-free.
    """
    start = np.radians([-278.0, 291.0])  # physically +82 / -69 deg
    target = np.radians([-10.0, 10.0])
    move = GotoMove(
        start_head_pose=np.eye(4),
        target_head_pose=None,
        start_antennas=start,
        target_antennas=target,
        start_body_yaw=0.0,
        target_body_yaw=None,
        duration=5.0,
        method=InterpolationTechnique.MIN_JERK,
    )
    traj = _evaluate_antennas(move, 5.0)

    # starts at the wrapped physical position, ends at the target
    assert np.allclose(traj[0], np.radians([82.0, -69.0]), atol=1e-6)
    assert np.allclose(traj[-1], target, atol=1e-6)
    # every command stays within one turn, with no per-step teleport
    assert np.all(np.abs(traj) <= np.pi + 1e-9)
    assert np.max(np.abs(np.diff(traj, axis=0))) < np.radians(5.0)


def test_goto_turn_free_antennas_unchanged():
    """Turn-free values (the normal case) are untouched by the wrap."""
    start = np.radians([-12.0, 10.0])
    target = np.radians([-175.0, 175.0])  # the sleep pose, near the boundary
    move = GotoMove(
        start_head_pose=np.eye(4),
        target_head_pose=None,
        start_antennas=start,
        target_antennas=target,
        start_body_yaw=0.0,
        target_body_yaw=None,
        duration=2.0,
        method=InterpolationTechnique.MIN_JERK,
    )
    traj = _evaluate_antennas(move, 2.0)
    assert np.allclose(traj[0], start, atol=1e-9)
    assert np.allclose(traj[-1], target, atol=1e-6)
    # direct path to the sleep antennas: no boundary crossing either
    assert np.max(np.abs(np.diff(traj, axis=0))) < np.radians(5.0)
