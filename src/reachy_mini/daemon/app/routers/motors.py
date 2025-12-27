"""Motors router.

Provides endpoints to get and set the motor control mode.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ....daemon.backend.abstract import Backend, MotorControlMode
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


class MotorModeRequest(BaseModel):
    """Optional request body for selective motor control.

    If motors is None or empty, all motors are affected.
    """

    motors: Optional[List[str]] = None


@router.get("/status")
async def get_motor_status(backend: Backend = Depends(get_backend)) -> MotorStatus:
    """Get the current status of the motors."""
    return MotorStatus(mode=backend.get_motor_control_mode())


@router.post("/set_mode/{mode}")
async def set_motor_mode(
    mode: MotorControlMode,
    request: Optional[MotorModeRequest] = None,
    backend: Backend = Depends(get_backend),
) -> dict[str, str]:
    """Set the motor control mode.

    Optionally pass JSON body with 'motors' list to affect only specific motors.
    Example: {"motors": ["left_antenna", "right_antenna"]}

    Available motor names: body_rotation, stewart_1-6, left_antenna, right_antenna
    """
    # If no specific motors requested, use global mode change
    if request is None or not request.motors:
        backend.set_motor_control_mode(mode)
        return {"status": f"all motors changed to {mode} mode"}

    # Selective motor control
    if mode not in [MotorControlMode.Enabled, MotorControlMode.Disabled]:
        raise HTTPException(
            status_code=400,
            detail=f"Selective motor control only supports 'enabled' or 'disabled', not '{mode}'"
        )

    try:
        backend.set_motor_torque_ids(request.motors, mode == MotorControlMode.Enabled)
        return {"status": f"motors {request.motors} changed to {mode} mode"}
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Unknown motor name: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
