"""Camera streaming API routes."""

import asyncio
from typing import AsyncGenerator

import cv2
import numpy as np
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from reachy_mini.media.camera_constants import CameraResolution
from reachy_mini.media.camera_opencv import OpenCVCamera

from ...backend.abstract import Backend
from ...backend.mujoco.backend import MujocoBackend
from ...daemon import Daemon
from ..dependencies import get_backend, get_daemon

router = APIRouter(prefix="/camera")

_shared_camera: OpenCVCamera | None = None
_camera_refs = 0


async def _get_shared_camera(is_sim: bool) -> OpenCVCamera:
    """Get or create shared camera instance."""
    global _shared_camera, _camera_refs
    if _shared_camera is None:
        _shared_camera = OpenCVCamera(log_level="WARNING", resolution=CameraResolution.R1280x720)
        _shared_camera.open(udp_camera="udp://@127.0.0.1:5005" if is_sim else None)
    _camera_refs += 1
    return _shared_camera


def _release_camera() -> None:
    """Release camera reference and close if no more clients."""
    global _shared_camera, _camera_refs
    _camera_refs -= 1
    if _camera_refs <= 0 and _shared_camera is not None:
        _shared_camera.close()
        _shared_camera = None
        _camera_refs = 0


@router.get("/stream")
async def stream_camera(
    backend: Backend = Depends(get_backend),
    daemon: Daemon = Depends(get_daemon),
) -> StreamingResponse:
    """Stream camera feed as MJPEG."""

    async def _stream() -> AsyncGenerator[bytes, None]:
        is_sim = bool(
            daemon.status().simulation_enabled and isinstance(backend, MujocoBackend)
        )
        cam = await _get_shared_camera(is_sim)
        try:
            while True:
                f = cam.read()
                if f is not None:
                    resized = np.asarray(cv2.resize(f, (640, 480)), dtype=np.uint8)
                    _, j = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if _:
                        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j.tobytes() + b"\r\n"
                await asyncio.sleep(0.04)
        finally:
            _release_camera()

    return StreamingResponse(_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

