"""Tests for pose interpolation, in particular the yaw_as_scalar routing fix."""

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import linear_pose_interpolation


def _world_yaw_deg(pose):
    return R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)[2]


def test_yaw_as_scalar_routes_through_front_not_back():
    """look-left(+120) -> look-right(-120) must route through 0, never the back (+-180).

    Regression for the SLERP geodesic taking the short 3D path around +-180 deg, which
    the bounded +-160 deg body yaw cannot follow (causing a discontinuous body_yaw flip).
    """
    start = create_head_pose(yaw=120, degrees=True)
    end = create_head_pose(yaw=-120, degrees=True)

    yaws = [
        _world_yaw_deg(
            linear_pose_interpolation(start, end, i / 60, yaw_as_scalar=True)
        )
        for i in range(61)
    ]

    # never swings out toward +-180 (the back)
    assert max(abs(y) for y in yaws) <= 121.0
    # passes through the front (~0 deg)
    assert min(abs(y) for y in yaws) <= 5.0
    # smooth: no large per-frame jump from a back crossover
    assert max(abs(b - a) for a, b in zip(yaws, yaws[1:])) < 10.0


def test_yaw_as_scalar_matches_slerp_without_back_crossing():
    """Outside the cross-the-back case, yaw_as_scalar is identical to the geodesic SLERP."""
    # same-side sweep with pitch/roll, body must move but no back crossing
    start = create_head_pose(yaw=130, pitch=15, roll=10, degrees=True)
    end = create_head_pose(yaw=70, pitch=15, roll=10, degrees=True)

    for i in range(21):
        t = i / 20
        np.testing.assert_allclose(
            linear_pose_interpolation(start, end, t, yaw_as_scalar=True),
            linear_pose_interpolation(start, end, t),
            atol=1e-9,
        )


def test_yaw_as_scalar_endpoints_exact():
    """t=0 and t=1 reproduce the start and target poses exactly."""
    start = create_head_pose(yaw=120, pitch=5, degrees=True)
    end = create_head_pose(yaw=-120, roll=8, degrees=True)

    np.testing.assert_allclose(
        linear_pose_interpolation(start, end, 0.0, yaw_as_scalar=True), start, atol=1e-9
    )
    np.testing.assert_allclose(
        linear_pose_interpolation(start, end, 1.0, yaw_as_scalar=True), end, atol=1e-9
    )
