"""XVF3800 audio-board configuration API routes.

Remote counterparts of `AudioBase.apply_audio_config()` and
`ReSpeaker.read_values()` (see `media/audio_control_utils.py`). The
on-robot daemon owns the USB handle, so these endpoints are the only
way for a remote (LAN or WebRTC) consumer to tune the audio board.

Each request opens a short-lived ReSpeaker USB handle, runs the call,
and closes — same pattern as `AudioBase.apply_audio_config`. The USB
control endpoints are stateless, so no shared handle is needed.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from reachy_mini.media.audio_control_utils import (
    AudioControlValue,
    init_respeaker_usb,
)

router = APIRouter(prefix="/audio/config")
logger = logging.getLogger(__name__)


class AudioParamPair(BaseModel):
    """One ``(parameter_name, values)`` pair in an audio config payload."""

    name: str
    values: list[float]


class ApplyAudioConfigRequest(BaseModel):
    """Request body for ``POST /audio/config/apply``."""

    config: list[AudioParamPair]
    verify: bool = True


class ApplyAudioConfigResponse(BaseModel):
    """Response body for ``POST /audio/config/apply``."""

    applied: bool


class ReadAudioParameterResponse(BaseModel):
    """Response body for ``GET /audio/config/parameter/{name}``."""

    name: str
    values: list[AudioControlValue]


@router.post("/apply")
async def apply_audio_config(req: ApplyAudioConfigRequest) -> ApplyAudioConfigResponse:
    """Write a batch of XVF3800 parameters and optionally verify them."""
    respeaker = init_respeaker_usb()
    if respeaker is None:
        raise HTTPException(
            status_code=503, detail="ReSpeaker audio board not available"
        )
    try:
        config = [(p.name, p.values) for p in req.config]
        applied = respeaker.apply_audio_config(config, verify=req.verify)
    except Exception as e:
        logger.exception("apply_audio_config failed")
        raise HTTPException(
            status_code=500, detail=f"apply_audio_config failed: {e}"
        ) from e
    finally:
        respeaker.close()
    return ApplyAudioConfigResponse(applied=applied)


@router.get("/parameter/{name}")
async def read_audio_parameter(name: str) -> ReadAudioParameterResponse:
    """Read a single XVF3800 parameter by name."""
    respeaker = init_respeaker_usb()
    if respeaker is None:
        raise HTTPException(
            status_code=503, detail="ReSpeaker audio board not available"
        )
    try:
        values = respeaker.read_values(name)
    except Exception as e:
        logger.exception("read_audio_parameter failed for %s", name)
        raise HTTPException(
            status_code=500, detail=f"read_audio_parameter failed: {e}"
        ) from e
    finally:
        respeaker.close()

    if values is None:
        raise HTTPException(
            status_code=404, detail=f"Parameter {name!r} could not be read"
        )
    return ReadAudioParameterResponse(name=name, values=list(values))
