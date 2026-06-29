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
from fastapi import APIRouter, HTTPException, Request
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


def _apply_device_change(http_request: Request, *, restart: bool) -> None:
    """Apply a device-selection change to the daemon's live audio and log how.

    ``restart=True`` (input device): the mic capture source is baked into the
    running media pipeline, so the pipeline is rebuilt to apply the change now
    (briefly interrupting any active audio/video stream). ``restart=False``
    (output device): the speaker sink is rebuilt on every ``play_sound``, so the
    change is used on the next sound — no restart needed.

    When an app currently holds the audio device (media released for direct
    access), the daemon's pipeline isn't running, so the change is only saved:
    it takes effect on the next app launch, or when the daemon re-acquires the
    audio hardware. An already-running app does NOT pick it up live.

    Best-effort: logged and ignored if the daemon is unavailable.
    """
    daemon = getattr(http_request.app.state, "daemon", None)
    if daemon is None:
        return

    if getattr(daemon, "media_released", False):
        logger.warning(
            "An app currently holds the audio device — the device change is saved "
            "but only takes effect on the next app launch (or when the daemon "
            "re-acquires the audio hardware). A running app will not pick it up live."
        )
        return

    if restart:
        logger.warning(
            "Restarting the media pipeline to apply the input device change now "
            "(this briefly interrupts any active audio/video stream)."
        )
        try:
            daemon.restart_media_pipeline()
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Could not restart media pipeline after device change: {e}")
    else:
        logger.info(
            "Output device change will be used the next time a sound is played."
        )


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
    request: SetDeviceRequest, http_request: Request
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

    changed = request.device_name != _selected_output_device
    _selected_output_device = request.device_name
    logger.info(f"Output device set to: {_selected_output_device}")
    if changed:
        _apply_device_change(http_request, restart=False)

    return SelectedDeviceResponse(device_name=_selected_output_device)


@router.delete("/output/selected")
async def clear_selected_output_device(http_request: Request) -> SelectedDeviceResponse:
    """Clear the selected output device (use default)."""
    global _selected_output_device
    changed = _selected_output_device is not None
    _selected_output_device = None
    logger.info("Output device cleared, using default")
    if changed:
        _apply_device_change(http_request, restart=False)

    return SelectedDeviceResponse(device_name=None)


@router.get("/input/selected")
async def get_selected_input_device() -> SelectedDeviceResponse:
    """Get the currently selected input device."""
    return SelectedDeviceResponse(device_name=_selected_input_device)


@router.post("/input/selected")
async def set_selected_input_device(
    request: SetDeviceRequest, http_request: Request
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

    changed = request.device_name != _selected_input_device
    _selected_input_device = request.device_name
    logger.info(f"Input device set to: {_selected_input_device}")
    if changed:
        _apply_device_change(http_request, restart=True)

    return SelectedDeviceResponse(device_name=_selected_input_device)


@router.delete("/input/selected")
async def clear_selected_input_device(http_request: Request) -> SelectedDeviceResponse:
    """Clear the selected input device (use default)."""
    global _selected_input_device
    changed = _selected_input_device is not None
    _selected_input_device = None
    logger.info("Input device cleared, using default")
    if changed:
        _apply_device_change(http_request, restart=True)

    return SelectedDeviceResponse(device_name=None)


# Helper functions for other modules


def get_local_selected_input() -> Optional[str]:
    """Return the in-process selected input device name (no HTTP).

    For daemon-internal consumers (volume control, media server) that run in
    the same process as this router. Reading the module state directly avoids a
    blocking self-HTTP call — calling the daemon's own API from within a request
    can't be served by the busy event loop and would stall until it times out.
    """
    return _selected_input_device


def get_local_selected_output() -> Optional[str]:
    """Return the in-process selected output device name (no HTTP).

    See :func:`get_local_selected_input` for why daemon-internal callers must
    not go through the HTTP helper.
    """
    return _selected_output_device


def get_selected_input() -> Optional[str]:
    """Get the currently selected input device name.

    For out-of-process clients (e.g. the SDK ``MediaManager``): tries the
    daemon API first, then falls back to local module state. Daemon-internal
    callers should use :func:`get_local_selected_input` instead.
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
