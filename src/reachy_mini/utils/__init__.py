"""Utility functions for Reachy Mini.

These functions provide various utilities such as creating head poses, performing minimum jerk interpolation,
checking if the Reachy Mini daemon is running, and performing linear pose interpolation.

"""

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R


def create_head_pose(
    x: float = 0,
    y: float = 0,
    z: float = 0,
    roll: float = 0,
    pitch: float = 0,
    yaw: float = 0,
    mm: bool = False,
    degrees: bool = True,
) -> npt.NDArray[np.float64]:
    """Create a homogeneous transformation matrix representing a pose in 6D space (position and orientation).

    Args:
        x (float): X coordinate of the position.
        y (float): Y coordinate of the position.
        z (float): Z coordinate of the position.
        roll (float): Roll angle
        pitch (float): Pitch angle
        yaw (float): Yaw angle
        mm (bool): If True, convert position from millimeters to meters.
        degrees (bool): If True, interpret roll, pitch, and yaw as degrees; otherwise as radians.

    Returns:
        np.ndarray: A 4x4 homogeneous transformation matrix representing the pose.

    """
    pose = np.eye(4)
    rot = R.from_euler("xyz", [roll, pitch, yaw], degrees=degrees).as_matrix()
    pose[:3, :3] = rot
    pose[:, 3] = [x, y, z, 1]
    if mm:
        pose[:3, 3] /= 1000

    return pose
