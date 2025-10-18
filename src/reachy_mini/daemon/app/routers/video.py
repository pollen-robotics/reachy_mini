"""Video streaming API routes."""

import time

from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse

from ...daemon import Daemon
from ...backend.mujoco.backend import MujocoBackend
from ..dependencies import get_daemon

router = APIRouter(
    prefix="/video",
)


def generate_mjpeg_stream(backend: MujocoBackend):
    """Generate MJPEG stream from MuJoCo backend.

    Yields frames in multipart MJPEG format for HTML streaming.
    """
    while True:
        # Get latest frame from backend
        with backend.viewer_frame_lock:
            frame_bytes = backend.latest_viewer_frame

        if frame_bytes is None:
            # No frame available yet, wait a bit
            time.sleep(0.05)
            continue

        # Yield frame in MJPEG format
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

        # Limit frame rate
        time.sleep(0.05)  # ~20 fps


@router.get("/stream")
async def stream_video(daemon: Daemon = Depends(get_daemon)) -> StreamingResponse:
    """Stream 3D viewer video as MJPEG.

    Returns:
        StreamingResponse: MJPEG stream for HTML <img> tag

    Raises:
        HTTPException: If daemon not running or not using MuJoCo backend
    """
    # Check if daemon is running
    if daemon.backend is None:
        return Response(
            content=b"Daemon not running",
            media_type="text/plain",
            status_code=503,
        )

    # Check if it's a MuJoCo backend
    if not isinstance(daemon.backend, MujocoBackend):
        return Response(
            content=b"Video streaming only available with MuJoCo simulator",
            media_type="text/plain",
            status_code=400,
        )

    return StreamingResponse(
        generate_mjpeg_stream(daemon.backend),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
