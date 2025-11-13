"""Volume control API routes.

This exposes:
- get current volume
- set volume
- play test sound
"""

import logging
import platform
import subprocess
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ....daemon.backend.abstract import Backend
from ..dependencies import get_backend

router = APIRouter(prefix="/volume")
logger = logging.getLogger(__name__)


class VolumeRequest(BaseModel):
    """Request model for setting volume."""

    volume: int = Field(..., ge=0, le=100, description="Volume level (0-100)")


class VolumeResponse(BaseModel):
    """Response model for volume operations."""

    volume: int
    device: str
    platform: str


def get_current_platform() -> str:
    """Get the current platform."""
    system = platform.system()
    if system == "Darwin":
        return "macOS"
    elif system == "Linux":
        return "Linux"
    else:
        return system


def detect_audio_device() -> str:
    """Detect the current audio output device."""
    system = platform.system()
    
    if system == "Linux":
        # Try to detect if Respeaker is available
        try:
            result = subprocess.run(
                ["aplay", "-l"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if "respeaker" in result.stdout.lower():
                return "respeaker"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return "default"
    elif system == "Darwin":
        return "system"
    else:
        return "unknown"


def get_volume_macos() -> Optional[int]:
    """Get current system volume on macOS."""
    try:
        result = subprocess.run(
            ["osascript", "-e", "output volume of (get volume settings)"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to get macOS volume: {e}")
    return None


def set_volume_macos(volume: int) -> bool:
    """Set system volume on macOS using osascript."""
    try:
        subprocess.run(
            ["osascript", "-e", f"set volume output volume {volume}"],
            capture_output=True,
            timeout=2,
            check=True,
        )
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.error(f"Failed to set macOS volume: {e}")
        return False


def get_volume_linux() -> Optional[int]:
    """Get current volume on Linux using amixer."""
    device_name = "Array" # Respeaker device
    try:
        cmd = ["bash", "-c", f"amixer -c {device_name} sget PCM | awk -F'[][]' '/Left:/ {{ print $2 }}'"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            # remove the % sign and convert to int
            return int(result.stdout.strip().rstrip("%")) 
                
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Failed to get linux volume: {e}")

    return None


def set_volume_linux(volume: int) -> bool:
    """Set current volume on Linux using amixer."""
    device_name = "Array"  # Respeaker device
    try:
        subprocess.run(
            ["amixer", "-c", device_name, "sset", "PCM", f"{volume}%"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.error(f"Failed to set Linux volume: {e}")
        return False


@router.get("/current")
async def get_volume() -> VolumeResponse:
    """Get the current volume level."""
    system = get_current_platform()
    device = detect_audio_device()
    
    volume = None
    if system == "macOS":
        volume = get_volume_macos()
    elif system == "Linux":
        volume = get_volume_linux()
    
    if volume is None:
        raise HTTPException(status_code=500, detail="Failed to get volume")
    
    return VolumeResponse(volume=volume, device=device, platform=system)


@router.post("/set")
async def set_volume(
    volume_req: VolumeRequest,
    backend: Backend = Depends(get_backend),
) -> VolumeResponse:
    """Set the volume level and play a test sound."""
    system = get_current_platform()
    device = detect_audio_device()
    
    success = False
    if system == "macOS":
        success = set_volume_macos(volume_req.volume)
    elif system == "Linux":
        success = set_volume_linux(volume_req.volume)
    else:
        raise HTTPException(
            status_code=501,
            detail=f"Volume control not supported on {system}",
        )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to set volume")
    
    # Play test sound
    try:
        test_sound = "impatient1.wav"
        if backend.audio:
            backend.audio.play_sound(test_sound, autoclean=True)
    except Exception as e:
        logger.warning(f"Failed to play test sound: {e}")
    
    return VolumeResponse(volume=volume_req.volume, device=device, platform=system)


@router.post("/test-sound")
async def play_test_sound(backend: Backend = Depends(get_backend)) -> dict[str, str]:
    """Play a test sound."""
    try:
        test_sound = "impatient1.wav"
        if backend.audio:
            backend.audio.play_sound(test_sound, autoclean=True)
            return {"status": "ok", "message": "Test sound played"}
        else:
            raise HTTPException(status_code=503, detail="Audio device not available")
    except Exception as e:
        logger.error(f"Failed to play test sound: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to play test sound: {e}")
