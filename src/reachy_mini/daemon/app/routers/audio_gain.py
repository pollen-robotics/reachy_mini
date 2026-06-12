"""Audio output gain API routes.

Exposes:
- GET  /api/audio/gain       — current output gain in dB
- POST /api/audio/gain       — set output gain (clamped to safe range)
"""

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from reachy_mini.daemon.app.dependencies import get_backend
from reachy_mini.daemon.backend.abstract import Backend
from reachy_mini.media.audio_gain import (
    MAX_GAIN_DB,
    MIN_GAIN_DB,
    db_to_linear,
    get_output_gain_db,
    set_output_gain_db,
)

router = APIRouter(prefix="/audio/gain")
logger = logging.getLogger(__name__)


class GainResponse(BaseModel):
    """Response model for gain operations."""

    gain_db: float
    gain_linear: float


class GainRequest(BaseModel):
    """Request model for setting gain."""

    gain_db: float = Field(
        ...,
        ge=MIN_GAIN_DB,
        le=MAX_GAIN_DB,
        description=f"Gain in dB ({MIN_GAIN_DB} to {MAX_GAIN_DB})",
    )


@router.get("/")
async def get_gain() -> GainResponse:
    """Get the current output gain level."""
    db = get_output_gain_db()
    return GainResponse(gain_db=db, gain_linear=db_to_linear(db))


@router.post("/")
async def set_gain(
    req: GainRequest,
    backend: Backend = Depends(get_backend),
) -> GainResponse:
    """Set the output gain level.

    The gain is applied immediately to all active audio pipelines.
    The value is clamped to the safe range and does not persist
    across daemon restarts (use REACHY_AUDIO_GAIN_DB env var for that).
    """
    clamped = set_output_gain_db(req.gain_db)
    linear = db_to_linear(clamped)

    # Update live GStreamer volume elements
    media_server = backend._media_server
    if media_server is not None:
        media_server.update_output_gain(linear)

    logger.info(f"Output gain set to {clamped:.1f} dB (linear {linear:.3f})")
    return GainResponse(gain_db=clamped, gain_linear=linear)
