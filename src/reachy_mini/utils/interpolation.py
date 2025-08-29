"""Interpolation utilities for Reachy Mini."""

from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
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


def delta_angle_between_mat_rot(
    P: NDArray[np.float64], Q: NDArray[np.float64]
) -> float:
    """Compute the angle (in radians) between two 3x3 rotation matrices `P` and `Q`.

    This is equivalent to the angular distance in axis-angle space.
    It is computed via the trace of the relative rotation matrix.

    References:
        - https://math.stackexchange.com/questions/2113634/comparing-two-rotation-matrices
        - http://www.boris-belousov.net/2016/12/01/quat-dist/

    Args:
        P: A 3x3 rotation matrix.
        Q: Another 3x3 rotation matrix.

    Returns:
        The angle in radians between the two rotations.

    """
    R = np.dot(P, Q.T)
    tr = (np.trace(R) - 1) / 2
    tr = np.clip(tr, -1.0, 1.0)  # Ensure numerical stability
    return np.arccos(tr)


def distance_between_poses(
    pose1: NDArray[np.float64], pose2: NDArray[np.float64]
) -> Tuple[float, float, float]:
    """Compute three types of distance between two 4x4 homogeneous transformation matrices.

    The result combines translation (in mm) and rotation (in degrees) using an arbitrary but
    emotionally satisfying equivalence: 1 degree ≈ 1 mm.

    Args:
        pose1: A 4x4 homogeneous transformation matrix representing the first pose.
        pose2: A 4x4 homogeneous transformation matrix representing the second pose.

    Returns:
        A tuple of:
        - translation distance in meters,
        - angular distance in radians,
        - unhinged distance in magic-mm (translation in mm + rotation in degrees).

    """
    distance_translation = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])
    distance_angle = delta_angle_between_mat_rot(pose1[:3, :3], pose2[:3, :3])
    unhinged_distance = distance_translation * 1000 + np.rad2deg(distance_angle)

    return distance_translation, distance_angle, unhinged_distance


def compose_world_offset(T_abs: np.ndarray, T_off_world: np.ndarray,
                         reorthonormalize: bool = False) -> np.ndarray:
    """
    Compose an absolute world-frame pose with a world-frame offset:
      - translations add in world:       t_final = t_abs + t_off
      - rotations compose in world:      R_final = R_off @ R_abs
    This rotates the frame in place (about its own origin) by a rotation
    defined in world axes, and shifts it by a world translation.

    Parameters
    ----------
    T_abs : (4,4) ndarray
        Absolute pose in world frame.
    T_off_world : (4,4) ndarray
        Offset transform specified in world axes (dx,dy,dz in world; dR about world axes).
    reorthonormalize : bool
        If True, SVD-orthonormalize the resulting rotation to fight drift.

    Returns
    -------
    T_final : (4,4) ndarray
        Resulting pose in world frame.
    """
    R_abs, t_abs = T_abs[:3, :3], T_abs[:3, 3]
    R_off, t_off = T_off_world[:3, :3], T_off_world[:3, 3]

    R_final = R_off @ R_abs
    if reorthonormalize:
        U, _, Vt = np.linalg.svd(R_final)
        R_final = U @ Vt

    t_final = t_abs + t_off

    T_final = np.eye(4)
    T_final[:3, :3] = R_final
    T_final[:3, 3]  = t_final
    return T_final
