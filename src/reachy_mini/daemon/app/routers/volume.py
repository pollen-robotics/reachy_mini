"""Volume control API routes.

This exposes:
- get current volume
- set volume
- same for microphone
- play test sound (optional)
"""

import logging
from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from reachy_mini.daemon.app.dependencies import get_backend
from reachy_mini.daemon.backend.abstract import Backend

from .volume_control import VolumeControl, create_volume_control

router = APIRouter(prefix="/volume")
logger = logging.getLogger(__name__)

# Lazily-initialised singleton – created on first request
# Remark: removing this could be an over-kill yet efficient way to account for dynamic changes in the audio devices :eyes:
_volume_control: VolumeControl | None = None


def _get_volume_control() -> VolumeControl:
    """Return the shared VolumeControl instance, creating it on first call."""
    global _volume_control  # noqa: PLW0603
    if _volume_control is None:
        _volume_control = create_volume_control()
    return _volume_control


class VolumeRequest(BaseModel):
    """Request model for setting volume."""

    volume: int = Field(..., ge=0, le=100, description="Volume level (0-100)")


class VolumeResponse(BaseModel):
    """Response model for volume operations."""

    volume: int
    platform: str
    device: str


class TestSoundResponse(BaseModel):
    """Response model for test sound operations."""

    status: str
    message: str


# ---- Helpers ----


def _read_volume(
    getter: Callable[[], int], vc: VolumeControl, device_name: str, error_detail: str
) -> VolumeResponse:
    """Read a volume value and return a VolumeResponse or raise on failure."""
    volume = getter()
    if volume < 0:
        raise HTTPException(status_code=500, detail=error_detail)
    return VolumeResponse(volume=volume, platform=vc.platform_name, device=device_name)


def _write_volume(
    setter: Callable[[int], bool],
    volume: int,
    vc: VolumeControl,
    device_name: str,
    error_detail: str,
) -> VolumeResponse:
    """Write a volume value and return a VolumeResponse or raise on failure."""
    if not setter(volume):
        raise HTTPException(status_code=500, detail=error_detail)
    return VolumeResponse(volume=volume, platform=vc.platform_name, device=device_name)


# ---- Speaker volume endpoints ----


@router.get("/current")
async def get_volume() -> VolumeResponse:
    """Get the current output volume level."""
    vc = _get_volume_control()
    return _read_volume(
        vc.get_output_volume, vc, vc.output_device.name, "Failed to get volume"
    )


@router.post("/set")
async def set_volume(
    volume_req: VolumeRequest,
    request: Request,
) -> VolumeResponse:
    """Set the output volume level and play a test sound."""
    vc = _get_volume_control()
    response = _write_volume(
        vc.set_output_volume,
        volume_req.volume,
        vc,
        vc.output_device.name,
        "Failed to set volume",
    )

    daemon = getattr(request.app.state, "daemon", None)
    backend: Backend | None = daemon.backend if daemon is not None else None
    if backend is not None and backend.ready.is_set():
        try:
            backend.play_sound("impatient1.wav")
        except Exception as e:
            logger.warning("Failed to play test sound: %s", e)

    return response


@router.post("/test-sound")
async def play_test_sound(backend: Backend = Depends(get_backend)) -> TestSoundResponse:
    """Play a test sound."""
    try:
        backend.play_sound("impatient1.wav")
        return TestSoundResponse(status="ok", message="Test sound played")
    except Exception as e:
        msg = str(e).lower()

        if "device unavailable" in msg or "-9985" in msg:
            logger.warning(
                "Test sound request while audio device is busy (likely GStreamer): %s",
                e,
            )
            return TestSoundResponse(
                status="busy",
                message="Audio device is currently in use, test sound was skipped.",
            )

        logger.error("Failed to play test sound: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to play test sound (see logs for details)",
        )


# ---- Microphone volume endpoints ----


@router.get("/microphone/current")
async def get_microphone_volume() -> VolumeResponse:
    """Get the current microphone input volume level."""
    vc = _get_volume_control()
    return _read_volume(
        vc.get_input_volume, vc, vc.input_device.name, "Failed to get microphone volume"
    )


@router.post("/microphone/set")
async def set_microphone_volume(volume_req: VolumeRequest) -> VolumeResponse:
    """Set the microphone input volume level."""
    vc = _get_volume_control()
    return _write_volume(
        vc.set_input_volume,
        volume_req.volume,
        vc,
        vc.input_device.name,
        "Failed to set microphone volume",
    )
