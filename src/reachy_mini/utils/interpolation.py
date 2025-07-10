"""Interpolation utilities for Reachy Mini."""

from typing import Callable, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

InterpolationFunc = Callable[[float], np.ndarray]


def minimum_jerk(
    starting_position: np.ndarray,
    goal_position: np.ndarray,
    duration: float,
    starting_velocity: Optional[np.ndarray] = None,
    starting_acceleration: Optional[np.ndarray] = None,
    final_velocity: Optional[np.ndarray] = None,
    final_acceleration: Optional[np.ndarray] = None,
) -> InterpolationFunc:
    """Compute the mimimum jerk interpolation function from starting position to goal position."""
    if starting_velocity is None:
        starting_velocity = np.zeros(starting_position.shape)
    if starting_acceleration is None:
        starting_acceleration = np.zeros(starting_position.shape)
    if final_velocity is None:
        final_velocity = np.zeros(goal_position.shape)
    if final_acceleration is None:
        final_acceleration = np.zeros(goal_position.shape)

    a0 = starting_position
    a1 = starting_velocity
    a2 = starting_acceleration / 2

    d1, d2, d3, d4, d5 = [duration**i for i in range(1, 6)]

    A = np.array(((d3, d4, d5), (3 * d2, 4 * d3, 5 * d4), (6 * d1, 12 * d2, 20 * d3)))
    B = np.array(
        (
            goal_position - a0 - (a1 * d1) - (a2 * d2),
            final_velocity - a1 - (2 * a2 * d1),
            final_acceleration - (2 * a2),
        )
    )
    X = np.linalg.solve(A, B)

    coeffs = [a0, a1, a2, X[0], X[1], X[2]]

    def f(t: float) -> np.ndarray:
        if t > duration:
            return goal_position
        return np.sum([c * t**i for i, c in enumerate(coeffs)], axis=0)

    return f


def linear_pose_interpolation(
    start_pose: np.ndarray, target_pose: np.ndarray, t: float
):
    """Linearly interpolate between two poses in 6D space."""
    # Extract rotations
    rot_start = R.from_matrix(start_pose[:3, :3])
    rot_end = R.from_matrix(target_pose[:3, :3])

    # Compute relative rotation q_rel such that rot_start * q_rel = rot_end
    q_rel = rot_start.inv() * rot_end
    # Convert to rotation vector (axis-angle)
    rotvec_rel = q_rel.as_rotvec()
    # Scale the rotation vector by t (allows t<0 or >1 for overshoot)
    rot_interp = (rot_start * R.from_rotvec(rotvec_rel * t)).as_matrix()

    # Extract translations
    pos_start = start_pose[:3, 3]
    pos_end = target_pose[:3, 3]
    # Linear interpolation/extrapolation on translation
    pos_interp = pos_start + (pos_end - pos_start) * t

    # Compose homogeneous transformation
    interp_pose = np.eye(4)
    interp_pose[:3, :3] = rot_interp
    interp_pose[:3, 3] = pos_interp

    return interp_pose


def time_trajectory(t: float, method="default"):
    """Compute the time trajectory value based on the specified interpolation method."""
    method = "minjerk" if method == "default" else method

    if t < 0 or t > 1:
        raise ValueError("time value is out of range [0,1]")

    if method == "linear":
        return t

    elif method == "minjerk":
        return 10 * t**3 - 15 * t**4 + 6 * t**5

    elif method == "ease":
        if t < 0.5:
            return 2 * t * t
        else:
            return 1 - ((-2 * t + 2) ** 2) / 2

    elif method == "cartoon":
        c1 = 1.70158
        c2 = c1 * 1.525

        if t < 0.5:
            # phase in
            return ((2 * t) ** 2 * ((c2 + 1) * 2 * t - c2)) / 2
        else:
            # phase out
            return (((2 * t - 2) ** 2 * ((c2 + 1) * (2 * t - 2) + c2)) + 2) / 2

    else:
        raise ValueError(
            "Unknown interpolation method: {} (possible values: linear, minjerk, ease, cartoon)".format(
                method
            )
        )
