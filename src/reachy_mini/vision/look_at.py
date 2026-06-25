"""Shared look-at geometry for camera-guided head aiming."""

import numpy as np
import numpy.typing as npt

from reachy_mini.media.camera_utils import undistort_points

DEFAULT_HEAD_TO_CAMERA_TRANSFORM: npt.NDArray[np.float64] = np.eye(4)
DEFAULT_HEAD_TO_CAMERA_TRANSFORM[:3, 3] = [0.0437, 0.0, 0.0512]
DEFAULT_HEAD_TO_CAMERA_TRANSFORM[:3, :3] = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)


def default_head_to_camera_transform() -> npt.NDArray[np.float64]:
    """Return the default transform from head frame to camera frame."""
    return DEFAULT_HEAD_TO_CAMERA_TRANSFORM.copy()


def _rotation_matrix_from_axis_angle(
    axis: npt.NDArray[np.float64],
    angle: float,
) -> npt.NDArray[np.float64]:
    """Build a 3x3 rotation matrix from a unit axis and an angle."""
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [
                c + x * x * one_minus_c,
                x * y * one_minus_c - z * s,
                x * z * one_minus_c + y * s,
            ],
            [
                y * x * one_minus_c + z * s,
                c + y * y * one_minus_c,
                y * z * one_minus_c - x * s,
            ],
            [
                z * x * one_minus_c - y * s,
                z * y * one_minus_c + x * s,
                c + z * z * one_minus_c,
            ],
        ],
        dtype=np.float64,
    )


def look_at_world_pose(x: float, y: float, z: float) -> npt.NDArray[np.float64]:
    """Return the head pose that points the head's +X axis at a world point."""
    target_position = np.array([x, y, z], dtype=np.float64)
    target_norm = np.linalg.norm(target_position)
    if target_norm < 1e-12:
        return np.eye(4, dtype=np.float64)

    target_vector = target_position / target_norm
    straight_head_vector = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    axis = np.cross(straight_head_vector, target_vector)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        if np.dot(straight_head_vector, target_vector) > 0:
            rot_mat = np.eye(3, dtype=np.float64)
        else:
            perp = (
                np.array([0.0, 1.0, 0.0], dtype=np.float64)
                if abs(straight_head_vector[0]) < 0.9
                else np.array([0.0, 0.0, 1.0], dtype=np.float64)
            )
            axis = np.cross(straight_head_vector, perp)
            axis /= np.linalg.norm(axis)
            rot_mat = _rotation_matrix_from_axis_angle(
                axis.astype(np.float64),
                np.pi,
            )
    else:
        axis /= axis_norm
        angle = np.arccos(
            np.clip(np.dot(straight_head_vector, target_vector), -1.0, 1.0)
        )
        rot_mat = _rotation_matrix_from_axis_angle(
            axis.astype(np.float64), float(angle)
        )

    target_head_pose = np.eye(4, dtype=np.float64)
    target_head_pose[:3, :3] = rot_mat
    return target_head_pose


def look_at_image_pose(
    u: float,
    v: float,
    K: npt.NDArray[np.float64],
    D: npt.NDArray[np.float64],
    T_world_head: npt.NDArray[np.float64],
    T_head_cam: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Return the head pose that looks at a pixel in the current camera frame."""
    if T_head_cam is None:
        T_head_cam = DEFAULT_HEAD_TO_CAMERA_TRANSFORM

    x_n, y_n = undistort_points(u, v, K, D)

    ray_cam = np.array([x_n, y_n, 1.0], dtype=np.float64)
    ray_cam /= np.linalg.norm(ray_cam)

    T_world_cam = T_world_head @ T_head_cam
    ray_world = T_world_cam[:3, :3] @ ray_cam
    p_world = T_world_cam[:3, 3] + ray_world

    return look_at_world_pose(
        x=float(p_world[0]),
        y=float(p_world[1]),
        z=float(p_world[2]),
    )
