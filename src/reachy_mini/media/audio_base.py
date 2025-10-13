"""Base classes for audio implementations.

The audio implementations support various backends and provide a unified
interface for audio input/output.
"""

import logging
import struct
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

import numpy as np
import numpy.typing as npt
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

    def __del__(self) -> None:
        """Destructor to ensure resources are released."""
        if self._respeaker:
            usb.util.dispose_resources(self._respeaker)

    @abstractmethod
    def start_recording(self) -> None:
        """Start recording audio."""
        pass

    @abstractmethod
    def get_audio_sample(self) -> Optional[bytes | npt.NDArray[np.float32]]:
        """Read audio data from the device. Returns the data or None if error."""
        pass

    @abstractmethod
    def get_audio_samplerate(self) -> int:
        """Return the samplerate of the audio device."""
        pass

    @abstractmethod
    def stop_recording(self) -> None:
        """Close the audio device and release resources."""
        pass

    @abstractmethod
    def start_playing(self) -> None:
        """Start playing audio."""
        pass

    @abstractmethod
    def push_audio_sample(self, data: bytes) -> None:
        """Push audio data to the output device."""
        pass

    @abstractmethod
    def stop_playing(self) -> None:
        """Stop playing audio and release resources."""
        pass

    @abstractmethod
    def play_sound(self, sound_file: str) -> None:
        """Play a sound file.

        Args:
            sound_file (str): Path to the sound file to play.

        """
        pass

    def _init_respeaker_usb(self) -> Optional[usb.core.Device]:
        dev = usb.core.find(idVendor=0x2886, idProduct=0x001A)
        if not dev:
            return None

        return dev

    def _read_usb(self, name: str) -> Optional[List[int] | List[float]]:
        try:
            data = self.PARAMETERS[name]
        except KeyError:
            self.logger.error(f"Unknown parameter: {name}")
            return None

        if not self._respeaker:
            self.logger.warning("ReSpeaker device not found.")
            return None

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

        result: Optional[List[float] | List[int]] = None
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
        if result is None:
            return None
        return int(result[1]), bool(result[3])
