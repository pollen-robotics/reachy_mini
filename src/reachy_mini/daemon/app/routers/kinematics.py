from fastapi import APIRouter, Depends

from ....daemon.backend.abstract import Backend
from ..dependencies import get_backend

router = APIRouter(
    prefix="/kinematics",
)


# Which kin solver for fk, ik: (placo, analytic, torch)
# params en plus automatic body yaw, check_collision


@router.get("/status")
async def get_kinematics_status():
    return {"status": {"kinematics": "enabled"}}


@router.get("/urdf")
async def get_urdf(backend: Backend = Depends(get_backend)):
    return {"urdf": backend.get_urdf()}
