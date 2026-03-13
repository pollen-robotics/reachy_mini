"""Media release/acquire API routes.

Allows clients to tell the daemon to release camera and audio hardware
for direct access (e.g. OpenCV, sounddevice), then re-acquire when done.
"""

from fastapi import APIRouter, Depends

from ...daemon import Daemon
from ..dependencies import get_daemon

router = APIRouter(
    prefix="/media",
)


@router.post("/release")
async def release_media(daemon: Daemon = Depends(get_daemon)) -> dict[str, str]:
    """Release camera and audio hardware for direct client access."""
    await daemon.release_media()
    return {"status": "ok"}


@router.post("/acquire")
async def acquire_media(daemon: Daemon = Depends(get_daemon)) -> dict[str, str]:
    """Re-acquire camera and audio hardware."""
    await daemon.acquire_media()
    return {"status": "ok"}


@router.get("/status")
async def media_status(daemon: Daemon = Depends(get_daemon)) -> dict[str, bool]:
    """Get the current media status."""
    return {
        "available": not daemon.media_released and daemon._media_server is not None,
        "released": daemon.media_released,
        "no_media": daemon.no_media,
    }
