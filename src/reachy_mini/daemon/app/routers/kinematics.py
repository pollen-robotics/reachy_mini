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
