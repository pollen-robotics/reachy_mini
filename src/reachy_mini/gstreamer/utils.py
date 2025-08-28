import logging
import subprocess
from enum import Enum

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst


class PlayerMode(Enum):
    WEBRTC = "webrtc"
    LOCAL = "local"


def get_respeaker_card_number() -> int:
    try:
        result = subprocess.run(
            ["arecord", "-l"], capture_output=True, text=True, check=True
        )
        output = result.stdout

        lines = output.split("\n")
        for line in lines:
            if "ReSpeaker" in line and "card" in line:
                card_number = line.split(":")[0].split("card ")[1].strip()
                logging.debug(f"Found ReSpeaker sound card: {card_number}")
                return int(card_number)

        logging.warning("ReSpeaker sound card not found. Returning default card")
        return 0  # default sound card

    except subprocess.CalledProcessError as e:
        logging.error(f"Cannot find sound card: {e}")
        return 0


def get_arducam_video_device() -> str:
    """
    Use Gst.DeviceMonitor to find the unix camera path /dev/videoX of the Arducam_12MP webcam.
    Returns the device path (e.g., '/dev/video2'), or '' if not found.
    """
    Gst.init(None)
    monitor = Gst.DeviceMonitor()
    monitor.add_filter("Video/Source")
    monitor.start()

    devices = monitor.get_devices()
    for device in devices:
        name = device.get_display_name()
        device_props = device.get_properties()
        if name and "Arducam_12MP" in name:
            if device_props and device_props.has_field("api.v4l2.path"):
                device_path = device_props.get_string("api.v4l2.path")
                logging.debug(f"Found Arducam_12MP at {device_path}")
                monitor.stop()
                return device_path
    monitor.stop()
    logging.warning("Arducam_12MP webcam not found.")
    return ""
