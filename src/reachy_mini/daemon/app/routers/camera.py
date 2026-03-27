"""Camera-related API routes.

Exposes camera specifications detected by the daemon so that clients
(SDK, REST, web UIs) can use the correct intrinsic matrix, distortion
coefficients, and available resolutions without hardcoding them.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from reachy_mini.media.camera_constants import get_camera_specs_by_name

from ...daemon import Daemon
from ..dependencies import get_daemon

router = APIRouter(prefix="/camera")


class ResolutionInfo(BaseModel):
    """A single camera resolution entry."""

    name: str
    width: int
    height: int
    fps: int
    crop_factor: float


class CameraSpecsResponse(BaseModel):
    """Full camera specifications as detected by the daemon."""

    name: str
    available_resolutions: list[ResolutionInfo]
    default_resolution: ResolutionInfo
    K: list[list[float]]
    D: list[float]


def _resolution_info(res) -> ResolutionInfo:  # type: ignore[no-untyped-def]
    """Convert a ``CameraResolution`` enum member to a ``ResolutionInfo``."""
    w, h, fps, crop = res.value
    return ResolutionInfo(
        name=res.name,
        width=w,
        height=h,
        fps=fps,
        crop_factor=crop,
    )


@router.get("/specs")
async def get_camera_specs(
    daemon: Daemon = Depends(get_daemon),
) -> CameraSpecsResponse:
    """Get the detected camera specifications.

    Returns the camera name, available resolutions, default resolution,
    intrinsic matrix (K) and distortion coefficients (D) for the camera
    that the daemon detected at startup.
    """
    specs = get_camera_specs_by_name(daemon.status().camera_specs_name)
    return CameraSpecsResponse(
        name=specs.name,
        available_resolutions=[
            _resolution_info(r) for r in specs.available_resolutions
        ],
        default_resolution=_resolution_info(specs.default_resolution),
        K=specs.K.tolist(),
        D=specs.D.tolist(),
    )
