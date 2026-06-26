"""Shared look-at geometry for camera-guided head aiming."""

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

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
            rot_mat = R.from_rotvec(np.pi * axis).as_matrix()
    else:
        axis /= axis_norm
        angle = np.arccos(
            np.clip(np.dot(straight_head_vector, target_vector), -1.0, 1.0)
        )
        rot_mat = R.from_rotvec(angle * axis).as_matrix()

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
