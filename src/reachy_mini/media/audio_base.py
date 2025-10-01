"""Base classes for audio implementations.

The audio implementations support various backends and provide a unified
interface for audio input/output.
"""

import logging
import struct
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import usb


class AudioBackend(Enum):
    """Audio backends."""

    SOUNDDEVICE = "sounddevice"
    GSTREAMER = "gstreamer"


class AudioBase(ABC):
    """Abstract class for opening and managing audio devices."""

    TIMEOUT = 100000
    PARAMETERS = {
        "VERSION": (48, 0, 4, "ro", "uint8"),
        "AEC_AZIMUTH_VALUES": (33, 75, 16 + 1, "ro", "radians"),
        "DOA_VALUE": (20, 18, 4 + 1, "ro", "uint16"),
    }

    def __init__(self, backend: AudioBackend, log_level: str = "INFO") -> None:
        """Initialize the audio device."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.backend = backend
        self._respeaker = self._init_respeaker_usb()
        # name, resid, cmdid, length, type

    def __del__(self):
        """Destructor to ensure resources are released."""
        if self._respeaker:
            usb.util.dispose_resources(self._respeaker)

    @abstractmethod
    def start_recording(self):
        """Start recording audio."""
        pass

    @abstractmethod
    def get_audio_sample(self):
        """Read audio data from the device. Returns the data or None if error."""
        pass

    @abstractmethod
    def get_audio_samplerate(self) -> int:
        """Return the samplerate of the audio device."""
        pass

    @abstractmethod
    def stop_recording(self):
        """Close the audio device and release resources."""
        pass

    @abstractmethod
    def start_playing(self):
        """Start playing audio."""
        pass

    @abstractmethod
    def push_audio_sample(self, data):
        """Push audio data to the output device."""
        pass

    @abstractmethod
    def stop_playing(self):
        """Stop playing audio and release resources."""
        pass

    def _init_respeaker_usb(self):
        dev = usb.core.find(idVendor=0x2886, idProduct=0x001A)
        if not dev:
            return None
        return dev

    def _read_usb(self, name):
        try:
            data = self.PARAMETERS[name]
        except KeyError:
            self.logger.error(f"Unknown parameter: {name}")
            return

        resid = data[0]
        cmdid = 0x80 | data[1]
        length = data[2]

        response = self._respeaker.ctrl_transfer(
            usb.util.CTRL_IN
            | usb.util.CTRL_TYPE_VENDOR
            | usb.util.CTRL_RECIPIENT_DEVICE,
            0,
            cmdid,
            resid,
            length,
            self.TIMEOUT,
        )

        self.logger.debug(f"Response for {name}: {response}")

        if data[4] == "uint8":
            result = response.tolist()
        elif data[4] == "radians":
            byte_data = response.tobytes()
            float1, float2, float3, float4 = struct.unpack("<ffff", byte_data[1:17])
            result = [
                np.rad2deg(float1),
                np.rad2deg(float2),
                np.rad2deg(float3),
                np.rad2deg(float4),
            ]
        elif data[4] == "uint16":
            result = response.tolist()

        return result

    def get_DoA(self) -> tuple[int, bool] | None:
        """Get the Direction of Arrival (DoA) value from the ReSpeaker device.

        0° is left, 90° is front/back, 180° is right

        Returns:
            tuple: A tuple containing the DoA value as an integer and the speech detection, or None if the device is not found.

        """
        if not self._respeaker:
            self.logger.warning("ReSpeaker device not found.")
            return None
        result = self._read_usb("DOA_VALUE")
        return result[1], bool(result[3])
