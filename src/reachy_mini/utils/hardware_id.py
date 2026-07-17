"""Robot-unique hardware identifier.

Reads the Pollen-branded audio device's USB serial number from sysfs.
This serial is burned in at manufacturing, unique per robot, and the
audio device is present on every Reachy Mini regardless of variant
(Lite has it on USB; Wireless has it on the CM4's internal USB bus).

The motor-bus USB-serial chip on Lite (CH343) also carries a serial,
but the motor bus on Wireless is wired directly to the CM4's UART
GPIO and never enumerates over USB — so the audio device is the only
hardware ID source that yields a single code path across both variants.

Externally the raw serial is *never* exposed: ``get_hardware_id()``
returns a SHA-256 hash truncated to 16 hex chars. The raw serial leaks
manufacturing batch info and is cross-referenceable with Pollen's
records, so we hide it from any client that doesn't own physical access
to the robot. The hash is deterministic, so it's still a stable
per-robot identifier suitable for fleet management, calibration cache
keys, and remote diagnostics. The BLE pairing PIN keeps deriving from
the raw serial — that's a different threat model (proximity-only,
short-lived secret).
"""

import hashlib
from pathlib import Path

POLLEN_AUDIO_VID = "38fb"
POLLEN_AUDIO_PID = "1001"


def _read_raw_audio_serial() -> str | None:
    """Read the raw Pollen audio device USB serial from sysfs (private)."""
    usb_root = Path("/sys/bus/usb/devices")
    if not usb_root.exists():
        return None
    for dev in usb_root.iterdir():
        try:
            if (dev / "idVendor").read_text().strip() != POLLEN_AUDIO_VID:
                continue
            if (dev / "idProduct").read_text().strip() != POLLEN_AUDIO_PID:
                continue
            serial = (dev / "serial").read_text().strip()
            return serial or None
        except (OSError, FileNotFoundError):
            continue
    return None


def get_hardware_id() -> str | None:
    """Return the robot's public hardware ID — a hash of the raw serial.

    SHA-256 of the raw audio device serial, truncated to the first 16
    hex chars (64 bits — collision-resistant for any plausible fleet).
    Stable per robot, deterministic, opaque. Returns ``None`` if no
    Reachy Mini is attached.
    """
    raw = _read_raw_audio_serial()
    if raw is None:
        return None
    return hashlib.sha256(raw.encode("ascii")).hexdigest()[:16]


def get_pin() -> str:
    """Return the 5-digit BLE pairing PIN derived from the raw serial.

    Uses the last 5 chars of the raw audio device serial. Pairing
    happens at BLE proximity, so leaking 5 chars of the raw serial
    via a PIN is acceptable; the public hardware ID stays hashed.
    Falls back to a fixed default when no robot is attached.
    """
    default_pin = "46879"
    raw = _read_raw_audio_serial()
    if raw and len(raw) >= 5:
        return raw[-5:]
    return default_pin
