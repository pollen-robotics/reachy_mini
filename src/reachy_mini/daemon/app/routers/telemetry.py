"""Read and change the user's anonymous telemetry choice."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from reachy_mini.daemon.telemetry import disabled_by_env, telemetry, user_enabled

router = APIRouter(prefix="/telemetry")


class TelemetrySettings(BaseModel):
    """User's telemetry choice."""

    enabled: bool


class TelemetryStatus(TelemetrySettings):
    """User's choice plus whether an environment variable overrides it."""

    disabled_by_env: bool


def _status() -> TelemetryStatus:
    return TelemetryStatus(enabled=user_enabled(), disabled_by_env=disabled_by_env())


@router.get("")
def get_telemetry() -> TelemetryStatus:
    """Get the telemetry choice. Nothing is sent while ``disabled_by_env``."""
    return _status()


@router.put("")
def set_telemetry(settings: TelemetrySettings) -> TelemetryStatus:
    """Enable or disable telemetry. Disabling also forgets the robot id."""
    try:
        telemetry.set_enabled(settings.enabled)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not save setting: {e}")
    return _status()
