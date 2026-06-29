"""Audio device selection API routes.

This exposes:
- list available input devices (microphones)
- list available output devices (speakers)
- get/set current input device
- get/set current output device
"""

import logging
from typing import Optional

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/audio-devices")
logger = logging.getLogger(__name__)

# Constants
DAEMON_BASE_URL = "http://127.0.0.1:8000"
DAEMON_API_TIMEOUT = 0.5  # Timeout in seconds for API calls

# In-memory storage for selected devices (persists until daemon restart)
_selected_input_device: Optional[str] = None
_selected_output_device: Optional[str] = None


class AudioDeviceListResponse(BaseModel):
    """Response model for listing audio devices."""

    devices: list[str]


class SelectedDeviceResponse(BaseModel):
    """Response model for the currently selected device."""

    device_name: Optional[str]


class SetDeviceRequest(BaseModel):
    """Request model for setting the audio device."""

    device_name: str


def get_device_names(device_class: str) -> list[str]:
    """List available audio device names via the shared GStreamer device monitor.

    Delegates enumeration and parsing to
    :func:`reachy_mini.media.device_detection.gst_monitor_devices` so device
    discovery lives in one place.

    Args:
        device_class: "Audio/Source" for microphones or "Audio/Sink" for speakers.

    """
    # Imported lazily so importing this router does not require GStreamer (gi).
    from reachy_mini.media.device_detection import gst_monitor_devices

    try:
        devices = gst_monitor_devices(device_class)
    except Exception as e:
        logger.error(f"Failed to list {device_class} devices: {e}")
        return []

    names: list[str] = []
    for device in devices:
        if device.display_name and device.display_name not in names:
            names.append(device.display_name)
    return names


# API Endpoints - List Devices


@router.get("/output")
async def get_output_devices() -> AudioDeviceListResponse:
    """List available audio output devices."""
    return AudioDeviceListResponse(devices=get_device_names("Audio/Sink"))


@router.get("/input")
async def get_input_devices() -> AudioDeviceListResponse:
    """List available audio input devices."""
    return AudioDeviceListResponse(devices=get_device_names("Audio/Source"))


# API Endpoints - Get/Set Selected Device


@router.get("/output/selected")
async def get_selected_output_device() -> SelectedDeviceResponse:
    """Get the currently selected output device."""
    return SelectedDeviceResponse(device_name=_selected_output_device)


@router.post("/output/selected")
async def set_selected_output_device(
    request: SetDeviceRequest,
) -> SelectedDeviceResponse:
    """Set the output device to use."""
    global _selected_output_device

    # Verify the device exists
    device_names = get_device_names("Audio/Sink")
    if request.device_name not in device_names:
        raise HTTPException(
            status_code=404,
            detail=f"Device '{request.device_name}' not found. Available: {device_names}",
        )

    _selected_output_device = request.device_name
    logger.info(f"Output device set to: {_selected_output_device}")
    logger.warning(
        "Audio output device selection updated; it will take effect on the next "
        "audio pipeline (re)start (e.g. new session or app launch)."
    )

    return SelectedDeviceResponse(device_name=_selected_output_device)


@router.delete("/output/selected")
async def clear_selected_output_device() -> SelectedDeviceResponse:
    """Clear the selected output device (use default)."""
    global _selected_output_device
    _selected_output_device = None
    logger.info("Output device cleared, using default")
    logger.warning(
        "Audio output device selection cleared; it will take effect on the next "
        "audio pipeline (re)start (e.g. new session or app launch)."
    )

    return SelectedDeviceResponse(device_name=None)


@router.get("/input/selected")
async def get_selected_input_device() -> SelectedDeviceResponse:
    """Get the currently selected input device."""
    return SelectedDeviceResponse(device_name=_selected_input_device)


@router.post("/input/selected")
async def set_selected_input_device(
    request: SetDeviceRequest,
) -> SelectedDeviceResponse:
    """Set the input device to use."""
    global _selected_input_device

    # Verify the device exists
    device_names = get_device_names("Audio/Source")
    if request.device_name not in device_names:
        raise HTTPException(
            status_code=404,
            detail=f"Device '{request.device_name}' not found. Available: {device_names}",
        )

    _selected_input_device = request.device_name
    logger.info(f"Input device set to: {_selected_input_device}")
    logger.warning(
        "Audio input device selection updated; it will take effect on the next "
        "audio pipeline (re)start (e.g. new session or app launch)."
    )

    return SelectedDeviceResponse(device_name=_selected_input_device)


@router.delete("/input/selected")
async def clear_selected_input_device() -> SelectedDeviceResponse:
    """Clear the selected input device (use default)."""
    global _selected_input_device
    _selected_input_device = None
    logger.info("Input device cleared, using default")
    logger.warning(
        "Audio input device selection cleared; it will take effect on the next "
        "audio pipeline (re)start (e.g. new session or app launch)."
    )

    return SelectedDeviceResponse(device_name=None)


# Helper functions for other modules


def get_selected_input() -> Optional[str]:
    """Get the currently selected input device name.

    Tries to fetch from the daemon API first, falls back to local config file.
    """
    try:
        response = requests.get(
            f"{DAEMON_BASE_URL}/api/audio-devices/input/selected",
            timeout=DAEMON_API_TIMEOUT,
        )
        if response.status_code == 200:
            data = response.json()
            device_name: str | None = data.get("device_name", None)
            return device_name
    except (requests.RequestException, ValueError) as e:
        logger.debug(f"Could not fetch input device from daemon API: {e}")

    # Fallback to local module variable (loaded from config file at import)
    return _selected_input_device


def get_selected_output() -> Optional[str]:
    """Get the currently selected output device name.

    Tries to fetch from the daemon API first, falls back to local config file.
    """
    try:
        response = requests.get(
            f"{DAEMON_BASE_URL}/api/audio-devices/output/selected",
            timeout=DAEMON_API_TIMEOUT,
        )
        if response.status_code == 200:
            data = response.json()
            device_name: str | None = data.get("device_name", None)
            return device_name
    except (requests.RequestException, ValueError) as e:
        logger.debug(f"Could not fetch output device from daemon API: {e}")

    # Fallback to local module variable (loaded from config file at import)
    return _selected_output_device
