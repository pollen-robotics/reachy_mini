"""Camera streaming API routes for real hardware camera."""

import time

import cv2
import numpy as np
from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse

from ...daemon import Daemon
from ..dependencies import get_daemon

router = APIRouter(
    prefix="/camera",
)


def generate_mjpeg_stream(daemon: Daemon):
    """Generate MJPEG stream from real camera via MediaManager.

    Yields frames in multipart MJPEG format for HTML streaming.
    Handles both OpenCV (numpy arrays) and GStreamer (JPEG bytes) backends.
    """
    while True:
        # Check if media_manager is available
        if daemon.media_manager is None:
            time.sleep(0.05)
            continue

        # Get latest frame from camera
        frame = daemon.media_manager.get_frame()

        if frame is None:
            # No frame available yet, wait a bit
            time.sleep(0.05)
            continue

        # Convert frame to JPEG bytes if needed
        if isinstance(frame, np.ndarray):
            # OpenCV backend returns numpy array, need to encode to JPEG
            success, jpeg_bytes = cv2.imencode('.jpg', frame)
            if not success:
                time.sleep(0.05)
                continue
            frame_bytes = jpeg_bytes.tobytes()
        else:
            # GStreamer backend returns JPEG bytes directly
            frame_bytes = frame

        # Yield frame in MJPEG format
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

        # Limit frame rate to ~20 fps
        time.sleep(0.05)


@router.get("/stream.mjpg")
async def stream_camera(daemon: Daemon = Depends(get_daemon)) -> StreamingResponse:
    """Stream real camera video as MJPEG.

    Returns:
        StreamingResponse: MJPEG stream for HTML <img> tag or OBS

    Raises:
        HTTPException: If daemon not running or camera not available
    """
    # Check if daemon is running
    if daemon.backend is None:
        return Response(
            content=b"Daemon not running",
            media_type="text/plain",
            status_code=503,
        )

    # Check if media manager is available
    if daemon.media_manager is None:
        return Response(
            content=b"Camera not available - MediaManager not initialized",
            media_type="text/plain",
            status_code=503,
        )

    return StreamingResponse(
        generate_mjpeg_stream(daemon),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
