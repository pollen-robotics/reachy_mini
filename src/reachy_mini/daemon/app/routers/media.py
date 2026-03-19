"""Media release/acquire API routes and remote sound management.

Allows clients to tell the daemon to release camera and audio hardware
for direct access (e.g. OpenCV, sounddevice), then re-acquire when done.

Also provides endpoints for remote sound playback and file management
so that WebRTC clients can upload, play, list and delete sound files on
the daemon.
"""

import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from ...daemon import Daemon
from ..dependencies import get_daemon

router = APIRouter(
    prefix="/media",
)

SOUNDS_TMP_DIR = "/tmp/reachy_mini_sounds"


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


class PlaySoundRequest(BaseModel):
    """Request body for the play_sound endpoint."""

    file: str


@router.post("/play_sound")
async def play_sound(
    body: PlaySoundRequest,
    daemon: Daemon = Depends(get_daemon),
) -> dict[str, str]:
    """Play a sound file on the robot's speaker.

    The *file* field can be:
    - An absolute path on the daemon's filesystem.
    - A filename relative to the built-in assets directory.
    - A filename previously uploaded to the sounds temp directory.
    """
    backend = daemon.backend
    if backend is None or not backend.ready.is_set():
        raise HTTPException(status_code=503, detail="Backend not running")

    # Resolve: if the filename lives in the temp upload directory, use
    # the full path so the backend can find it.
    sound_file = body.file
    if not os.path.isabs(sound_file):
        tmp_candidate = os.path.join(SOUNDS_TMP_DIR, sound_file)
        if os.path.isfile(tmp_candidate):
            sound_file = tmp_candidate

    backend.play_sound(sound_file)
    return {"status": "ok"}


@router.post("/stop_sound")
async def stop_sound(
    daemon: Daemon = Depends(get_daemon),
) -> dict[str, str]:
    """Stop the currently playing sound file."""
    backend = daemon.backend
    if backend is None or not backend.ready.is_set():
        raise HTTPException(status_code=503, detail="Backend not running")

    backend.stop_sound()
    return {"status": "ok"}


@router.post("/sounds/upload")
async def upload_sound(
    file: UploadFile = File(...),
) -> dict[str, str]:
    """Upload a sound file to the daemon's temporary sound directory.

    The file is saved to ``/tmp/reachy_mini_sounds/<original_filename>``.
    If a file with the same name already exists it is overwritten.

    Returns:
        JSON with the absolute *path* of the saved file on the daemon.

    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # Reject path traversal
    filename = Path(file.filename).name
    if not filename or filename in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")

    os.makedirs(SOUNDS_TMP_DIR, exist_ok=True)
    dest = os.path.join(SOUNDS_TMP_DIR, filename)

    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    return {"status": "ok", "path": dest}


@router.get("/sounds")
async def list_sounds() -> dict[str, list[str]]:
    """List sound files in the daemon's temporary sound directory."""
    if not os.path.isdir(SOUNDS_TMP_DIR):
        return {"files": []}
    files = sorted(
        entry.name for entry in os.scandir(SOUNDS_TMP_DIR) if entry.is_file()
    )
    return {"files": files}


@router.delete("/sounds/{filename}")
async def delete_sound(filename: str) -> dict[str, str]:
    """Delete a sound file from the daemon's temporary sound directory.

    Only files inside the temp directory can be deleted (no path traversal).
    """
    # Reject path traversal
    safe_name = Path(filename).name
    if not safe_name or safe_name != filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    filepath = os.path.join(SOUNDS_TMP_DIR, safe_name)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

    os.remove(filepath)
    return {"status": "ok"}
