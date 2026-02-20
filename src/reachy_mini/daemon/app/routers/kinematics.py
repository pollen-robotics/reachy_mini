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
from ..models import FullBodyTarget

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


@router.get("/limits")
async def get_kinematics_limits(
    backend: Backend = Depends(get_backend),
) -> dict[str, Any]:
    """Get the task-space motion limits of the robot."""
    return {"limits": backend.get_limits()}


@router.post("/is_reachable")
async def check_reachable(
    target: FullBodyTarget,
    backend: Backend = Depends(get_backend),
) -> dict[str, bool]:
    """Check if a target pose is within the reachable workspace."""
    if target.target_head_pose is None:
        raise HTTPException(
            status_code=422,
            detail="target_head_pose is required",
        )
    pose = target.target_head_pose.to_pose_array()
    body_yaw = target.target_body_yaw if target.target_body_yaw is not None else 0.0
    reachable = backend.is_pose_reachable(pose, body_yaw)
    return {"reachable": reachable}


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
