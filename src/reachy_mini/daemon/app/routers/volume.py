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
    test_sound = "impatient1.wav"
    if backend.audio:
        try:
            backend.audio.play_sound(test_sound, autoclean=True)
        except Exception as e:
            msg = str(e).lower()
            if "device unavailable" in msg or "-9985" in msg:
                logger.warning(
                    "Test sound not played: audio device busy (likely GStreamer): %s",
                    e,
                )
            else:
                logger.warning("Failed to play test sound: %s", e)
    else:
        logger.warning("No audio backend available, skipping test sound.")

    return VolumeResponse(volume=volume_req.volume, device=device, platform=system)


@router.post("/test-sound")
async def play_test_sound(backend: Backend = Depends(get_backend)) -> dict[str, str]:
    """Play a test sound."""
    test_sound = "impatient1.wav"

    if not backend.audio:
        raise HTTPException(status_code=503, detail="Audio device not available")

    try:
        backend.audio.play_sound(test_sound, autoclean=True)
        return {"status": "ok", "message": "Test sound played"}
    except Exception as e:
        msg = str(e).lower()

        # Special-case ALSA / PortAudio device-busy situation
        if "device unavailable" in msg or "-9985" in msg:
            logger.warning(
                "Test sound request while audio device is busy (likely GStreamer): %s",
                e,
            )
            # Still 200, but tell the caller it was skipped
            return {
                "status": "busy",
                "message": "Audio device is currently in use, test sound was skipped.",
            }

        # Any other error is treated as a real failure
        logger.error("Failed to play test sound: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to play test sound (see logs for details)",
        )


# Microphone control endpoints

def get_microphone_volume_macos() -> Optional[int]:
    """Get current microphone input volume on macOS."""
    try:
        result = subprocess.run(
            ["osascript", "-e", "input volume of (get volume settings)"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to get macOS microphone volume: {e}")
    return None


def set_microphone_volume_macos(volume: int) -> bool:
    """Set microphone input volume on macOS using osascript."""
    try:
        subprocess.run(
            ["osascript", "-e", f"set volume input volume {volume}"],
            capture_output=True,
            timeout=2,
            check=True,
        )
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.error(f"Failed to set macOS microphone volume: {e}")
        return False


def get_microphone_volume_linux() -> Optional[int]:
    """Get current microphone input volume on Linux using amixer."""
    device_name = "Array"  # Respeaker device
    try:
        cmd = ["bash", "-c", f"amixer -c {device_name} sget Capture | awk -F'[][]' '/Left:/ {{ print $2 }}'"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            # remove the % sign and convert to int
            return int(result.stdout.strip().rstrip("%"))
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to get Linux microphone volume: {e}")
    return None


def set_microphone_volume_linux(volume: int) -> bool:
    """Set microphone input volume on Linux using amixer."""
    device_name = "Array"  # Respeaker device
    try:
        subprocess.run(
            ["amixer", "-c", device_name, "sset", "Capture", f"{volume}%"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.error(f"Failed to set Linux microphone volume: {e}")
        return False


@router.get("/microphone/current")
async def get_microphone_volume() -> VolumeResponse:
    """Get the current microphone input volume level."""
    system = get_current_platform()
    device = detect_audio_device()
    
    volume = None
    if system == "macOS":
        volume = get_microphone_volume_macos()
    elif system == "Linux":
        volume = get_microphone_volume_linux()
    
    if volume is None:
        raise HTTPException(status_code=500, detail="Failed to get microphone volume")
    
    return VolumeResponse(volume=volume, device=device, platform=system)


@router.post("/microphone/set")
async def set_microphone_volume(
    volume_req: VolumeRequest,
) -> VolumeResponse:
    """Set the microphone input volume level."""
    system = get_current_platform()
    device = detect_audio_device()
    
    success = False
    if system == "macOS":
        success = set_microphone_volume_macos(volume_req.volume)
    elif system == "Linux":
        success = set_microphone_volume_linux(volume_req.volume)
    else:
        raise HTTPException(
            status_code=501,
            detail=f"Microphone volume control not supported on {system}",
        )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to set microphone volume")
    
    return VolumeResponse(volume=volume_req.volume, device=device, platform=system)
