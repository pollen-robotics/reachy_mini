from enum import Enum

from fastapi import APIRouter, Depends

from ....daemon.backend.abstract import Backend
from ..dependencies import get_backend


class MotorMode(str, Enum):
    Enabled = "enabled"
    Disabled = "disabled"


router = APIRouter(
    prefix="/motors",
)


@router.post("/set_mode/{mode}")
async def set_motor_mode(mode: MotorMode, backend: Backend = Depends(get_backend)):
    if mode == MotorMode.Enabled:
        backend.enable_motors()
    elif mode == MotorMode.Disabled:
        backend.disable_motors()
    else:
        raise ValueError(f"Unknown motor mode: {mode}")
    return {"status": f"motors changed to {mode} mode"}
