"""Audio device selection API routes: list, get, and set input/output devices."""

import logging

import requests
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/audio-devices")
logger = logging.getLogger(__name__)

DAEMON_BASE_URL = "http://127.0.0.1:8000"
DAEMON_API_TIMEOUT = 0.5

# Selected devices live in memory only; they reset when the daemon restarts.
_selected_input_device: str | None = None
_selected_output_device: str | None = None


class AudioDeviceListResponse(BaseModel):
    """Response model for listing audio devices."""

    devices: list[str]


class SelectedDeviceResponse(BaseModel):
    """Response model for the currently selected device."""

    device_name: str | None


class SetDeviceRequest(BaseModel):
    """Request model for setting the audio device."""

    device_name: str


def get_device_names(device_class: str) -> list[str]:
    """List device names for a GStreamer class (Audio/Source or Audio/Sink)."""
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
    """Apply a device-selection change to the daemon's live audio (best-effort).

    ``restart=True`` (input) rebuilds the media pipeline now, briefly
    interrupting any stream; ``restart=False`` (output) is picked up on the
    next sound. No-op (change only saved) when an app holds the audio device.
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


@router.get("/output")
async def get_output_devices() -> AudioDeviceListResponse:
    """List available audio output devices."""
    return AudioDeviceListResponse(devices=get_device_names("Audio/Sink"))


@router.get("/input")
async def get_input_devices() -> AudioDeviceListResponse:
    """List available audio input devices."""
    return AudioDeviceListResponse(devices=get_device_names("Audio/Source"))


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


def get_local_selected_input() -> str | None:
    """Return the selected input device name from in-process state (no HTTP).

    For daemon-internal callers: a self-HTTP call would stall the busy event loop.
    """
    return _selected_input_device


def get_local_selected_output() -> str | None:
    """Return the selected output device name from in-process state (no HTTP).

    See :func:`get_local_selected_input` for why daemon-internal callers avoid HTTP.
    """
    return _selected_output_device


def get_selected_input() -> str | None:
    """Get the selected input device name (for out-of-process SDK clients).

    Tries the daemon API, then falls back to module state; daemon-internal
    callers should use :func:`get_local_selected_input`.
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

    return _selected_input_device


def get_selected_output() -> str | None:
    """Get the selected output device name (for out-of-process SDK clients).

    Tries the daemon API, then falls back to module state; daemon-internal
    callers should use :func:`get_local_selected_output`.
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

    return _selected_output_device
