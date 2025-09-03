from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ....daemon.backend.abstract import Backend, MotorControlMode
from ..dependencies import get_backend

router = APIRouter(
    prefix="/motors",
)


class MotorStatus(BaseModel):
    mode: MotorControlMode


@router.get("/status")
async def get_motor_status(backend: Backend = Depends(get_backend)) -> MotorStatus:
    return MotorStatus(mode=backend.get_motor_control_mode())


@router.post("/set_mode/{mode}")
async def set_motor_mode(
    mode: MotorControlMode,
    backend: Backend = Depends(get_backend),
):
    backend.set_motor_control_mode(mode)

    return {"status": f"motors changed to {mode} mode"}
