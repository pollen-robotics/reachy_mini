"""Camera streaming endpoints for the Reachy Mini daemon.

Provides endpoints for getting camera frames:
- GET /api/camera/frame - Single JPEG frame
- GET /api/camera/stream - MJPEG stream (for <img> tags)
"""

import asyncio
import io
import logging
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, StreamingResponse

from reachy_mini.daemon.daemon import Daemon

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/camera", tags=["camera"])


def get_daemon(request: Request) -> Daemon:
    """Get the daemon from the app state."""
    return request.app.state.daemon


@router.get("/frame")
async def get_frame(
    quality: int = 80,
    daemon: Daemon = Depends(get_daemon),
) -> Response:
    """Get a single JPEG frame from the camera.
    
    Args:
        quality: JPEG quality (1-100, default 80)
        
    Returns:
        JPEG image response
    """
    if daemon.media_manager is None:
        return Response(
            content=b"Camera not available",
            status_code=503,
            media_type="text/plain",
        )
    
    frame = daemon.media_manager.get_frame()
    
    if frame is None:
        return Response(
            content=b"No frame available",
            status_code=503,
            media_type="text/plain",
        )
    
    # Encode frame as JPEG
    quality = max(1, min(100, quality))
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    
    return Response(
        content=jpeg.tobytes(),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


async def generate_mjpeg_stream(daemon: Daemon, fps: int = 15, quality: int = 70):
    """Generate MJPEG stream frames.
    
    Args:
        daemon: The daemon instance
        fps: Target frames per second
        quality: JPEG quality (1-100)
    """
    period = 1.0 / fps
    quality = max(1, min(100, quality))
    
    while True:
        try:
            if daemon.media_manager is None:
                await asyncio.sleep(1)
                continue
            
            frame = daemon.media_manager.get_frame()
            
            if frame is not None:
                # Encode frame as JPEG
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                jpeg_bytes = jpeg.tobytes()
                
                # MJPEG boundary format
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n"
                    b"\r\n" + jpeg_bytes + b"\r\n"
                )
            
            await asyncio.sleep(period)
            
        except Exception as e:
            logger.error(f"Error generating MJPEG frame: {e}")
            await asyncio.sleep(1)


@router.get("/stream")
async def get_stream(
    fps: int = 15,
    quality: int = 70,
    daemon: Daemon = Depends(get_daemon),
) -> StreamingResponse:
    """Get MJPEG video stream from the camera.
    
    This endpoint returns a multipart/x-mixed-replace stream that can be
    used directly in an <img> tag for live video.
    
    Args:
        fps: Target frames per second (default 15)
        quality: JPEG quality 1-100 (default 70)
        
    Returns:
        MJPEG streaming response
        
    Example:
        <img src="http://localhost:8000/api/camera/stream" />
    """
    return StreamingResponse(
        generate_mjpeg_stream(daemon, fps=fps, quality=quality),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*",
        },
    )

