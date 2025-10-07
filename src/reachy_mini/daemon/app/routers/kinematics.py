"""Kinematics router for handling kinematics-related requests.

This module defines the API endpoints for interacting with the kinematics
subsystem of the robot. It provides endpoints for retrieving URDF representation,
and other kinematics-related information.
"""

from typing import Any

from fastapi import APIRouter, Depends

from ....daemon.backend.abstract import Backend
from ..dependencies import get_backend

router = APIRouter(
    prefix="/kinematics",
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
