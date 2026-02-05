"""Motor monitoring router - provides access to motor current and temperature data."""

from fastapi import APIRouter, Depends

from ....daemon.backend.abstract import Backend
from ....daemon.backend.robot.backend import RobotBackend
from ..dependencies import get_backend

router = APIRouter(
    prefix="/motor_monitoring",
)


@router.get("/stewart_current")
async def get_stewart_current(backend: Backend = Depends(get_backend)) -> dict:
    """Get current readings for Stewart platform motors."""
    if not isinstance(backend, RobotBackend):
        return {"error": "Not available in simulation mode", "currents": [0] * 6}

    try:
        currents = backend.c.read_stewart_platform_current()
        return {
            "currents_ma": list(currents),
            "motor_names": [
                "stewart_1",
                "stewart_2",
                "stewart_3",
                "stewart_4",
                "stewart_5",
                "stewart_6",
            ],
        }
    except Exception as e:
        return {"error": str(e), "currents": [0] * 6}
