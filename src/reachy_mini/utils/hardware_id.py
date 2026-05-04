"""Robot-unique hardware identifier.

Reads the Pollen-branded audio device's USB serial number from sysfs.
This serial is burned in at manufacturing, unique per robot, and the
audio device is present on every Reachy Mini regardless of variant
(Lite has it on USB; Wireless has it on the CM4's internal USB bus).

The motor-bus USB-serial chip on Lite (CH343) also carries a serial,
but the motor bus on Wireless is wired directly to the CM4's UART
GPIO and never enumerates over USB — so the audio device is the only
hardware ID source that yields a single code path across both variants.
"""

from pathlib import Path

POLLEN_AUDIO_VID = "38fb"
POLLEN_AUDIO_PID = "1001"


def get_hardware_id() -> str | None:
    """Return the robot's unique hardware ID, or ``None`` if not found.

    The ID is the USB serial number of the Pollen audio device
    (VID=0x38fb, PID=0x1001). Read directly from sysfs; works as a
    normal user, no DFU mode required, no third-party tools.
    """
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


def get_pin() -> str:
    """Return the 5-digit BLE pairing PIN derived from the hardware ID.

    Uses the last 5 chars of ``get_hardware_id()``. Falls back to a
    fixed default when the audio device is not detected (e.g. running
    on a developer machine without a robot attached).
    """
    default_pin = "46879"
    hwid = get_hardware_id()
    if hwid and len(hwid) >= 5:
        return hwid[-5:]
    return default_pin
