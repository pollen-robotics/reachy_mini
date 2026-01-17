"""Audio device selection API routes.

This exposes:
- list available input devices (microphones)
- list available output devices (speakers)
- get/set current input device
- get/set current output device
"""

import logging
import re
import subprocess
from enum import Enum
from typing import Callable, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/audio-devices")
logger = logging.getLogger(__name__)

# Constants
AUDIO_COMMAND_TIMEOUT = 2  # Timeout in seconds for audio commands

# Callback type: takes device_name (str or None) as argument
DeviceChangeCallback = Callable[[Optional[str]], None]


class DeviceType(Enum):
    """Audio device type."""

    INPUT = "input"
    OUTPUT = "output"


# In-memory storage for selected devices (persists until daemon restart)
_selected_input_device: Optional[str] = None
_selected_output_device: Optional[str] = None

# Callbacks triggered when device selection changes
_on_input_device_changed: Optional[DeviceChangeCallback] = None
_on_output_device_changed: Optional[DeviceChangeCallback] = None


class AudioDeviceListResponse(BaseModel):
    """Response model for listing audio devices."""

    devices: list[str]


class SelectedDeviceResponse(BaseModel):
    """Response model for the currently selected device."""

    device_name: Optional[str]


class SetDeviceRequest(BaseModel):
    """Request model for setting the audio device."""

    device_name: str


def parse_device_names(output: str) -> list[str]:
    """Parse the output of aplay -l or arecord -l to extract device names.

    Example output format:
        **** List of PLAYBACK Hardware Devices ****
        card 0: Headphones [bcm2835 Headphones], device 0: bcm2835 Headphones [bcm2835 Headphones]
          Subdevices: 8/8
          Subdevice #0: subdevice #0
        card 1: Audio [Reachy Mini Audio], device 0: USB Audio [USB Audio]
          Subdevices: 1/1
          Subdevice #0: subdevice #0
    """
    names = []
    # Match lines like: card 0: Headphones [bcm2835 Headphones], device 0: ...
    pattern = r"card \d+: \w+ \[([^\]]+)\], device \d+:"

    for line in output.split("\n"):
        match = re.search(pattern, line)
        if match:
            name = match.group(1)
            if name not in names:
                names.append(name)

    return names


def get_device_names(device_type: DeviceType) -> list[str]:
    """List available audio device names on Linux.

    Args:
        device_type: DeviceType.INPUT for microphones (arecord -l) or DeviceType.OUTPUT for speakers (aplay -l)
    """
    cmd = ["aplay", "-l"] if device_type == DeviceType.OUTPUT else ["arecord", "-l"]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=AUDIO_COMMAND_TIMEOUT,
        )
        if result.returncode == 0:
            return parse_device_names(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Failed to list {device_type.value} devices: {e}")
    return []


# API Endpoints - List Devices


@router.get("/output")
async def get_output_devices() -> AudioDeviceListResponse:
    """List available audio output devices."""
    return AudioDeviceListResponse(devices=get_device_names(DeviceType.OUTPUT))


@router.get("/input")
async def get_input_devices() -> AudioDeviceListResponse:
    """List available audio input devices."""
    return AudioDeviceListResponse(devices=get_device_names(DeviceType.INPUT))


# API Endpoints - Get/Set Selected Device


@router.get("/output/selected")
async def get_selected_output_device() -> SelectedDeviceResponse:
    """Get the currently selected output device."""
    return SelectedDeviceResponse(device_name=_selected_output_device)


@router.post("/output/selected")
async def set_selected_output_device(request: SetDeviceRequest) -> SelectedDeviceResponse:
    """Set the output device to use."""
    global _selected_output_device

    # Verify the device exists
    device_names = get_device_names(DeviceType.OUTPUT)
    if request.device_name not in device_names:
        raise HTTPException(
            status_code=404,
            detail=f"Device '{request.device_name}' not found. Available: {device_names}",
        )

    _selected_output_device = request.device_name
    logger.info(f"Output device set to: {_selected_output_device}")

    if _on_output_device_changed:
        _on_output_device_changed(_selected_output_device)

    return SelectedDeviceResponse(device_name=_selected_output_device)


@router.delete("/output/selected")
async def clear_selected_output_device() -> SelectedDeviceResponse:
    """Clear the selected output device (use default)."""
    global _selected_output_device
    _selected_output_device = None
    logger.info("Output device cleared, using default")

    if _on_output_device_changed:
        _on_output_device_changed(None)

    return SelectedDeviceResponse(device_name=None)


@router.get("/input/selected")
async def get_selected_input_device() -> SelectedDeviceResponse:
    """Get the currently selected input device."""
    return SelectedDeviceResponse(device_name=_selected_input_device)


@router.post("/input/selected")
async def set_selected_input_device(request: SetDeviceRequest) -> SelectedDeviceResponse:
    """Set the input device to use."""
    global _selected_input_device

    # Verify the device exists
    device_names = get_device_names(DeviceType.INPUT)
    if request.device_name not in device_names:
        raise HTTPException(
            status_code=404,
            detail=f"Device '{request.device_name}' not found. Available: {device_names}",
        )

    _selected_input_device = request.device_name
    logger.info(f"Input device set to: {_selected_input_device}")

    if _on_input_device_changed:
        _on_input_device_changed(_selected_input_device)

    return SelectedDeviceResponse(device_name=_selected_input_device)


@router.delete("/input/selected")
async def clear_selected_input_device() -> SelectedDeviceResponse:
    """Clear the selected input device (use default)."""
    global _selected_input_device
    _selected_input_device = None
    logger.info("Input device cleared, using default")

    if _on_input_device_changed:
        _on_input_device_changed(None)

    return SelectedDeviceResponse(device_name=None)


# Helper functions for other modules


def get_selected_input() -> Optional[str]:
    """Get the currently selected input device name."""
    return _selected_input_device


def get_selected_output() -> Optional[str]:
    """Get the currently selected output device name."""
    return _selected_output_device


def on_input_device_changed(callback: DeviceChangeCallback) -> None:
    """Register a callback to be called when the input device changes.

    Args:
        callback: Function that takes the new device name (or None) as argument.

    Example:
        def handle_input_change(device_name: str | None):
            print(f"Input device changed to: {device_name}")
            # Restart audio pipeline with new device...

        on_input_device_changed(handle_input_change)
    """
    global _on_input_device_changed
    _on_input_device_changed = callback


def on_output_device_changed(callback: DeviceChangeCallback) -> None:
    """Register a callback to be called when the output device changes.

    Args:
        callback: Function that takes the new device name (or None) as argument.

    Example:
        def handle_output_change(device_name: str | None):
            print(f"Output device changed to: {device_name}")
            # Restart audio pipeline with new device...

        on_output_device_changed(handle_output_change)
    """
    global _on_output_device_changed
    _on_output_device_changed = callback
