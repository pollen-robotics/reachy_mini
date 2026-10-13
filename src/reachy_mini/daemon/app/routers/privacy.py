"""Privacy mode API routes.

Privacy mode cuts the camera and the microphones for every consumer (see
``reachy_mini.daemon.privacy``). These routes drive the ``"api"`` source; the
NFC privacy hat is a second, independent one.
"""

import asyncio

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ...daemon import Daemon
from ..dependencies import get_daemon

router = APIRouter(prefix="/privacy")


class PrivacyRequest(BaseModel):
    """Request body to switch privacy mode on or off."""

    enabled: bool


class PrivacyStatus(BaseModel):
    """Whether privacy mode is on, and who is asking for it."""

    enabled: bool
    sources: list[str]


@router.get("")
async def get_privacy(daemon: Daemon = Depends(get_daemon)) -> PrivacyStatus:
    """Get the privacy mode status."""
    return PrivacyStatus(enabled=daemon.privacy.enabled, sources=daemon.privacy.sources)


@router.post("")
async def set_privacy(
    request: PrivacyRequest, daemon: Daemon = Depends(get_daemon)
) -> PrivacyStatus:
    """Switch privacy mode on or off.

    Switching it off only withdraws this API's request: the answer still says
    ``enabled`` while a privacy hat is on the robot's head.
    """
    # Muting the microphones is a USB round trip: keep it off the event loop.
    await asyncio.to_thread(daemon.privacy.set, "api", request.enabled)
    return PrivacyStatus(enabled=daemon.privacy.enabled, sources=daemon.privacy.sources)
