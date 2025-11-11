"""Kinematics router for handling kinematics-related requests.

This module defines the API endpoints for interacting with the kinematics
subsystem of the robot. It provides endpoints for retrieving URDF representation,
and other kinematics-related information.
"""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response

from ....daemon.backend.abstract import Backend
from ..dependencies import get_backend

router = APIRouter(
    prefix="/kinematics",
)

STL_ASSETS_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "descriptions"
    / "reachy_mini"
    / "urdf"
    / "assets"
)


@router.get("/info")
async def get_kinematics_info(
    backend: Backend = Depends(get_backend),
) -> dict[str, Any]:
    """Get the current information of the kinematics."""
    return {
        "info": {
            "engine": backend.kinematics_engine,
            "collision check": backend.check_collision,
        }
    }


@router.get("/urdf")
async def get_urdf(backend: Backend = Depends(get_backend)) -> dict[str, str]:
    """Get the URDF representation of the robot."""
    return {"urdf": backend.get_urdf()}


@router.get("/stl/{filename}")
async def get_stl_file(filename: Path) -> Response:
    """Get the path to an STL asset file."""
    file_path = STL_ASSETS_DIR / filename
    try:
        with open(file_path, "rb") as file:
            content = file.read()
            return Response(content, media_type="model/stl")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"STL file not found {file_path}")


@router.get("/look_at_image")
async def look_at_image(
    u: int,
    v: int,
    use_pose_matrix: bool = False,
    backend: Backend = Depends(get_backend),
) -> Any:
    """Calculate head pose to look at a pixel position in camera image.

    Args:
        u: Horizontal pixel coordinate
        v: Vertical pixel coordinate
        use_pose_matrix: Return Matrix4x4Pose if True, else XYZRPYPose

    Returns:
        Target head pose for looking at the specified pixel
    """
    import cv2
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    from ..models import as_any_pose

    # Camera calibration parameters (from ReachyMini SDK)
    K = np.array([[550.3564, 0.0, 638.0112],
                  [0.0, 549.1653, 364.589],
                  [0.0, 0.0, 1.0]])
    D = np.array([-0.0694, 0.1565, -0.0004, 0.0003, -0.0983])

    # Head to camera transform
    T_head_cam = np.eye(4)
    T_head_cam[:3, 3] = [0.0437, 0, 0.0512]
    T_head_cam[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])

    # Undistort pixel coordinates
    x_n, y_n = cv2.undistortPoints(np.float32([[[u, v]]]), K, D)[0, 0]

    # Convert to 3D ray in camera frame
    ray_cam = np.array([x_n, y_n, 1.0])
    ray_cam /= np.linalg.norm(ray_cam)

    # Get current head pose and transform ray to world coordinates
    T_world_head = backend.get_present_head_pose()
    T_world_cam = T_world_head @ T_head_cam

    R_wc = T_world_cam[:3, :3]
    t_wc = T_world_cam[:3, 3]
    ray_world = R_wc @ ray_cam
    P_world = t_wc + ray_world

    # Calculate rotation to look at target point
    target_vector = P_world / np.linalg.norm(P_world)
    straight_head_vector = np.array([1, 0, 0])

    v1 = straight_head_vector
    v2 = target_vector
    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-8:
        # Vectors are parallel
        if np.dot(v1, v2) > 0:
            rot_mat = np.eye(3)
        else:
            # Opposite direction: rotate 180° around perpendicular axis
            perp = np.array([0, 1, 0]) if abs(v1[0]) < 0.9 else np.array([0, 0, 1])
            axis = np.cross(v1, perp)
            axis /= np.linalg.norm(axis)
            rot_mat = R.from_rotvec(np.pi * axis).as_matrix()
    else:
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        rotation_vector = angle * axis
        rot_mat = R.from_rotvec(rotation_vector).as_matrix()

    # Build target pose
    target_head_pose = np.eye(4)
    target_head_pose[:3, :3] = rot_mat

    return as_any_pose(target_head_pose, use_pose_matrix)
