"""Motors router.

Provides endpoints to get and set the motor control mode.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from reachy_mini.io.protocol import MotorControlMode

from ....daemon.backend.abstract import Backend
from ..dependencies import get_backend

router = APIRouter(
    prefix="/motors",
)


class MotorStatus(BaseModel):
    """Represents the status of the motors.

    Exposes
    - mode: The current motor control mode (enabled, disabled, gravity_compensation).
    """

    mode: MotorControlMode


@router.get("/status")
async def get_motor_status(backend: Backend = Depends(get_backend)) -> MotorStatus:
    """Get the current status of the motors."""
    return MotorStatus(mode=backend.get_motor_control_mode())


@router.post("/set_mode/{mode}")
async def set_motor_mode(
    mode: MotorControlMode,
    backend: Backend = Depends(get_backend),
) -> dict[str, str]:
    """Set the motor control mode."""
    try:
        backend.set_motor_control_mode(mode)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return {"status": f"motors changed to {mode} mode"}
