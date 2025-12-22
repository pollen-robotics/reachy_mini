"""Motors router.

Provides endpoints to get and set the motor control mode.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials
from pydantic import BaseModel

from ....daemon.backend.abstract import Backend, MotorControlMode
from ..dependencies import get_backend

router = APIRouter(
    prefix="/motors",
)

security = HTTPBearer(scheme_name="Authorization")


def verify_auth(credentials: HTTPAuthCredentials = Depends(security)) -> HTTPAuthCredentials:
    """Verify authentication credentials for protected endpoints.
    
    This is a security check to prevent unauthorized access to critical
    motor control endpoints. Any request without a valid Bearer token will be rejected.
    """
    if not credentials or not credentials.credentials.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials


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
    _: None = Depends(verify_auth),
) -> dict[str, str]:
    """Set the motor control mode.
    
    Requires Bearer token authentication to prevent unauthorized motor control.
    """
    backend.set_motor_control_mode(mode)

    return {"status": f"motors changed to {mode} mode"}
