"""GStreamer device detection utilities.

Provides pure-logic functions for finding Reachy Mini audio and video
devices from a list of :class:`DeviceInfo` descriptors.  The detection
logic operates on plain data so it can be unit-tested without hardware.

A text parser (:func:`parse_gst_device_monitor_output`) is also included
so that captured ``gst-device-monitor-1.0`` output can feed the same
functions in tests.
"""

from __future__ import annotations

import logging
import platform
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import gi

from reachy_mini.media.camera_constants import (
    ArducamSpecs,
    CameraSpecs,
    ReachyMiniLiteCamSpecs,
    ReachyMiniWirelessCamSpecs,
)

gi.require_version("Gst", "1.0")

from gi.repository import Gst  # noqa: E402

_logger = logging.getLogger(__name__)

# Default camera name priority order used by ``find_video_device``.
DEFAULT_CAM_NAMES: Sequence[str] = ("Reachy", "Arducam_12MP", "imx708")

# Default target name for audio device detection.
DEFAULT_AUDIO_TARGET: str = "Reachy Mini Audio"


@dataclass
class DeviceInfo:
    """Lightweight, GStreamer-free representation of a device.

    Attributes:
        display_name: Human-readable device name reported by GStreamer
            (e.g. ``"Reachy Mini Audio Analog Stereo"``).
        device_class: GStreamer device class string
            (e.g. ``"Audio/Source"``, ``"Video/Source"``).
        properties: Flat mapping of all GStreamer structure fields,
            with every value stored as a string.
        index: Position of the device in the monitor's device list.
            Relevant on macOS where the ``avfvideosrc`` element uses the
            index to select a camera.

    """

    display_name: str
    device_class: str
    properties: dict[str, str] = field(default_factory=dict)
    index: int = 0


# Matches "Device found:" block header.
_DEVICE_HEADER_RE = re.compile(r"^Device found:\s*$")
# Matches the "name  : …" line inside a block.
_NAME_RE = re.compile(r"^\s+name\s+:\s+(.+)$")
# Matches the "class : …" line.
_CLASS_RE = re.compile(r"^\s+class\s+:\s+(.+)$")
# Matches a "key = value" property line (one leading tab + two-tab indent).
_PROP_RE = re.compile(r"^\s+(\S+)\s+=\s+(.*)$")


def parse_gst_device_monitor_output(text: str) -> list[DeviceInfo]:
    """Parse captured ``gst-device-monitor-1.0`` text into device descriptors.

    The parser recognises ``Device found:`` blocks and extracts the
    ``name``, ``class``, and all ``key = value`` property lines.
    Capability (``caps``) lines and ``gst-launch`` hint lines are
    silently ignored.

    Args:
        text: Full text output of ``gst-device-monitor-1.0``.

    Returns:
        An ordered list of :class:`DeviceInfo` instances, one per
        ``Device found:`` block, preserving their original order.

    """
    devices: list[DeviceInfo] = []
    lines = text.splitlines()

    idx = 0
    device_counter = 0
    while idx < len(lines):
        line = lines[idx]

        if not _DEVICE_HEADER_RE.match(line):
            idx += 1
            continue

        # Start of a new device block.
        name = ""
        cls = ""
        props: dict[str, str] = {}
        idx += 1

        in_properties = False
        while idx < len(lines):
            line = lines[idx]

            # Empty line between devices — end of block.
            if line.strip() == "" and (name or cls or props):
                # Only break if we've actually parsed something.
                # Multiple blank lines before content are skipped.
                if name or cls:
                    break

            # gst-launch hint line — marks end of the properties section.
            if line.strip().startswith("gst-launch"):
                idx += 1
                break

            m_name = _NAME_RE.match(line)
            if m_name:
                name = m_name.group(1).strip()
                idx += 1
                continue

            m_class = _CLASS_RE.match(line)
            if m_class:
                cls = m_class.group(1).strip()
                idx += 1
                continue

            if line.strip() == "properties:":
                in_properties = True
                idx += 1
                continue

            if in_properties:
                m_prop = _PROP_RE.match(line)
                if m_prop:
                    key = m_prop.group(1)
                    value = m_prop.group(2).strip()
                    # Strip trailing type hints like ``(gboolean)``
                    # and surrounding quotes from PulseAudio-style escaping.
                    value = re.sub(r"\s*\([^)]*\)\s*$", "", value)
                    value = value.strip('"').replace("\\ ", " ")
                    props[key] = value
                elif line.startswith("\t\t") or line.startswith("        "):
                    # Continuation of caps or multi-line value — skip.
                    pass
                else:
                    # No longer inside properties.
                    in_properties = False

            idx += 1

        if name or cls:
            devices.append(
                DeviceInfo(
                    display_name=name,
                    device_class=cls,
                    properties=props,
                    index=device_counter,
                )
            )
            device_counter += 1

    return devices


def gst_structure_get_field(structure: Any, field_name: str) -> Any:
    """Safely extract a single field value from a ``Gst.Structure``.

    Handles ``GstValueArray`` / ``GstValueList`` fields that
    ``get_value()`` cannot auto-unbox, and falls back to ``None``
    for any other unsupported GStreamer fundamental type.
    """
    field_type = structure.get_field_type(field_name)
    if field_type == Gst.ValueArray.__gtype__:
        ok, arr = structure.get_array(field_name)
        if ok and arr is not None:
            return [arr.get_nth(j) for j in range(arr.n_values)]
        return None
    if field_type == Gst.ValueList.__gtype__:
        ok, arr = structure.get_list(field_name)
        if ok and arr is not None:
            return [arr.get_nth(j) for j in range(arr.n_values)]
        return None
    try:
        return structure.get_value(field_name)
    except TypeError:
        return None


def gst_props_to_dict(props: Gst.Structure) -> dict[str, str]:
    """Convert all fields of a ``Gst.Structure`` to a flat string dict."""
    output_dict: dict[str, str] = {}
    for i in range(props.n_fields()):
        field_name: str = props.nth_field_name(i)
        value = gst_structure_get_field(props, field_name)
        output_dict[field_name] = str(value) if value is not None else ""
    return output_dict


def gst_device_to_device_info(device: Any, index: int = 0) -> DeviceInfo:
    """Convert a ``Gst.Device`` to a :class:`DeviceInfo`.

    Reads the display name, device class, and iterates all structure
    fields to build a flat ``dict[str, str]`` of properties.

    Args:
        device: A ``Gst.Device`` instance (typed as ``Any`` because
            GStreamer does not ship type stubs).
        index: Position in the device list (used on macOS).

    Returns:
        A :class:`DeviceInfo` with all available properties.

    """
    name: str = device.get_display_name() or ""
    device_class: str = device.get_device_class() or ""

    gst_props = device.get_properties()
    props = gst_props_to_dict(gst_props) if gst_props is not None else {}

    return DeviceInfo(
        display_name=name,
        device_class=device_class,
        properties=props,
        index=index,
    )


def gst_devices_to_device_infos(gst_devices: Any) -> list[DeviceInfo]:
    """Convert a list of ``Gst.Device`` objects to :class:`DeviceInfo` objects.

    Args:
        gst_devices: Iterable of ``Gst.Device`` instances.

    Returns:
        List of :class:`DeviceInfo`, preserving order and index.

    """
    return [gst_device_to_device_info(d, index=i) for i, d in enumerate(gst_devices)]


def gst_monitor_devices(filter_class: str) -> list[DeviceInfo]:
    """Start a ``Gst.DeviceMonitor``, query devices, and return :class:`DeviceInfo` list.

    This is the only function in this module that actually imports and uses
    GStreamer.  It is meant to be called by the media server / audio classes
    as a thin wrapper around the monitor lifecycle.

    Args:
        filter_class: GStreamer device class filter, e.g.
            ``"Audio/Source"`` or ``"Video/Source"``.

    Returns:
        List of :class:`DeviceInfo` for all devices matching the filter.

    Raises:
        Exception: Propagates any GStreamer error.

    """
    monitor = Gst.DeviceMonitor()
    monitor.add_filter(filter_class)
    monitor.start()
    try:
        return gst_devices_to_device_infos(monitor.get_devices())
    finally:
        monitor.stop()


def find_audio_device(
    devices: List[DeviceInfo],
    device_type: str,
    current_platform: str | None = None,
    target_name: str = DEFAULT_AUDIO_TARGET,
) -> Optional[str]:
    """Find the Reachy Mini audio device identifier from a device list.

    Iterates *devices* looking for one whose :attr:`DeviceInfo.display_name`
    contains *target_name*.  Monitor and loopback sources are skipped.
    The returned identifier is platform-specific:

    * **PipeWire** — ``node.name`` property.
    * **Windows WASAPI** — ``device.id`` property (non-loopback).
    * **macOS CoreAudio** — ``unique-id`` property.
    * **Linux PulseAudio** — constructed
      ``alsa_{input|output}.<udev.id>.<profile>`` string.
    * **Linux ALSA fallback** — ``device.string`` property.

    Args:
        devices: Device descriptors (from :func:`gst_devices_to_device_infos`
            or :func:`parse_gst_device_monitor_output`).
        device_type: ``"Source"`` for microphone or ``"Sink"`` for speaker.
        current_platform: Value of :func:`platform.system` (e.g.
            ``"Linux"``, ``"Windows"``, ``"Darwin"``).  Defaults to the
            current host platform when ``None``.
        target_name: Substring to match in display names.

    Returns:
        The platform-specific device identifier, or ``None`` if no
        matching device is found.

    """
    if current_platform is None:
        current_platform = platform.system()

    for device in devices:
        if target_name not in device.display_name:
            continue

        props = device.properties

        match current_platform:
            case "Linux":
                # Skip PulseAudio monitor sources (e.g. "Monitor of …").
                if props.get("device.class") == "monitor":
                    continue
                # PipeWire exposes node.name; preferred when available.
                if "node.name" in props:
                    node_name = props["node.name"]
                    _logger.debug("Found audio %s device: %s", device_type, node_name)
                    return node_name
                # PulseAudio fallback: construct from udev.id + profile.
                udev_id = props.get("udev.id")
                profile = props.get("device.profile.name")
                if udev_id and profile:
                    prefix = "alsa_output" if device_type == "Sink" else "alsa_input"
                    pa_device = f"{prefix}.{udev_id}.{profile}"
                    _logger.debug(
                        "Found audio %s device (PulseAudio): %s",
                        device_type,
                        pa_device,
                    )
                    return pa_device
                # Raw ALSA fallback.
                if "device.string" in props:
                    device_id = props["device.string"]
                    _logger.debug(
                        "Found audio %s device (ALSA): %s",
                        device_type,
                        device_id,
                    )
                    return device_id
            case "Windows":
                if props.get("device.api") != "wasapi2":
                    continue
                # Skip loopback capture devices when looking for a real source.
                # Use .lower() because the live GStreamer API returns Python bools
                # whose str() representation is "True"/"False" (capital), while the
                # text dump from gst-device-monitor uses lowercase "true"/"false".
                if (
                    device_type == "Source"
                    and props.get("wasapi2.device.loopback", "false").lower() == "true"
                ):
                    continue
                device_id = props.get("device.id", "")
                _logger.debug(
                    "Found audio %s device (WASAPI): %s", device_type, device_id
                )
                return device_id
            case "Darwin":
                if "unique-id" in props:
                    device_id = props["unique-id"]
                    _logger.debug(
                        "Found audio %s device (CoreAudio): %s",
                        device_type,
                        device_id,
                    )
                    return device_id

    _logger.warning("No %s %s card found.", target_name, device_type)
    return None


def _make_camera_specs(cam_name: str) -> CameraSpecs:
    """Return the appropriate ``CameraSpecs`` instance for a camera name."""
    if cam_name == "Arducam_12MP":
        return ArducamSpecs()
    return ReachyMiniLiteCamSpecs()


def find_video_device(
    devices: List[DeviceInfo],
    current_platform: str | None = None,
    cam_names: Sequence[str] = DEFAULT_CAM_NAMES,
) -> Tuple[str, Optional[CameraSpecs]]:
    """Find the Reachy Mini camera from a device list.

    Camera names are tried in priority order (default:
    ``Reachy`` > ``Arducam_12MP`` > ``imx708``).  The first match wins.

    The returned path string is platform-specific:

    * **Linux V4L2** — ``/dev/videoN`` from ``api.v4l2.path``.
    * **RPi CSI (imx708)** — the literal string ``"imx708"``.
    * **Windows** — the display name (for ``mfvideosrc``).
    * **macOS** — the device index as a string (for ``avfvideosrc``).

    Args:
        devices: Device descriptors.
        current_platform: Value of :func:`platform.system`.
            Defaults to the current host platform when ``None``.
        cam_names: Camera name substrings to search for, in priority order.

    Returns:
        A ``(device_path, camera_specs)`` tuple.  ``device_path`` is
        ``""`` and ``camera_specs`` is ``None`` when no camera is found.

    """
    if current_platform is None:
        current_platform = platform.system()

    for cam_name in cam_names:
        for device in devices:
            if cam_name not in device.display_name:
                continue

            props = device.properties

            match current_platform:
                case "Linux":
                    if "api.v4l2.path" in props:
                        device_path = props["api.v4l2.path"]
                        _logger.debug("Found %s camera at %s", cam_name, device_path)
                        return device_path, _make_camera_specs(cam_name)
                    elif cam_name == "imx708":
                        _logger.debug("Found %s camera (CSI)", cam_name)
                        return cam_name, ReachyMiniWirelessCamSpecs()
                case "Windows":
                    _logger.debug(
                        "Found %s camera on Windows: %s",
                        cam_name,
                        device.display_name,
                    )
                    return device.display_name, _make_camera_specs(cam_name)
                case "Darwin":
                    _logger.debug(
                        "Found %s camera on macOS at index %d",
                        cam_name,
                        device.index,
                    )
                    return str(device.index), _make_camera_specs(cam_name)

    _logger.warning("No camera found.")
    return "", None


def get_audio_device(device_type: str = "Source") -> Optional[str]:
    """Detect the Reachy Mini audio device via ``Gst.DeviceMonitor``.

    This is the high-level entry point that both ``GstMediaServer`` and
    ``GStreamerAudio`` should call.  It handles the full monitor
    lifecycle (start / query / stop) and delegates to
    :func:`find_audio_device` for the actual matching.

    Args:
        device_type: ``"Source"`` for microphone or ``"Sink"`` for speaker.

    Returns:
        The platform-specific device identifier, or ``None``.

    """
    try:
        devices = gst_monitor_devices(f"Audio/{device_type}")
        return find_audio_device(devices, device_type)
    except Exception:
        _logger.exception("Error detecting audio %s device", device_type)
        return None


def get_video_device() -> Tuple[str, Optional[CameraSpecs]]:
    """Detect the Reachy Mini camera via ``Gst.DeviceMonitor``.

    Handles the full monitor lifecycle and delegates to
    :func:`find_video_device` for matching.

    Returns:
        A ``(device_path, camera_specs)`` tuple.  ``device_path`` is
        ``""`` and ``camera_specs`` is ``None`` when no camera is found.

    """
    try:
        devices = gst_monitor_devices("Video/Source")
        return find_video_device(devices)
    except Exception:
        _logger.exception("Error detecting video device")
        return "", None
