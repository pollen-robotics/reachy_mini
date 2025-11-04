"""Base classes for audio implementations.

The audio implementations support various backends and provide a unified
interface for audio input/output.
"""

import logging
import struct
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import usb
from libusb_package import get_libusb1_backend


class AudioBase(ABC):
    """Abstract class for opening and managing audio devices."""

    SAMPLE_RATE = 16000  # respeaker samplerate
    TIMEOUT = 100000
    PARAMETERS = {
        "VERSION": (48, 0, 4, "ro", "uint8"),
        "AEC_AZIMUTH_VALUES": (33, 75, 16 + 1, "ro", "radians"),
        "DOA_VALUE": (20, 18, 4 + 1, "ro", "uint16"),
        "DOA_VALUE_RADIANS": (20, 19, 8 + 1, "ro", "radians"),
    }

    def __init__(self, log_level: str = "INFO") -> None:
        """Initialize the audio device."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
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
    def get_audio_sample(self) -> Optional[npt.NDArray[np.float32]]:
        """Read audio data from the device. Returns the data or None if error."""
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
    def push_audio_sample(self, data: npt.NDArray[np.float32]) -> None:
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
        try:
            dev = usb.core.find(
                idVendor=0x2886, idProduct=0x001A, backend=get_libusb1_backend()
            )
            return dev
        except usb.core.NoBackendError:
            self.logger.error(
                "No USB backend was found ! Make sure libusb_package is correctly installed with `pip install libusb_package`."
            )
            return None

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
            num_values = (data[2] - 1) / 4
            match_str = "<"
            for i in range(int(num_values)):
                match_str += "f"
            result = [
                float(x) for x in struct.unpack(match_str, byte_data[1 : data[2]])
            ]
        elif data[4] == "uint16":
            result = response.tolist()

        return result

    def get_DoA(self) -> tuple[float, bool] | None:
        """Get the Direction of Arrival (DoA) value from the ReSpeaker device.

        0° is left, 90° is front/back, 180° is right

        Returns:
            tuple: A tuple containing the DoA value as an integer and the speech detection, or None if the device is not found.

        """
        if not self._respeaker:
            self.logger.warning("ReSpeaker device not found.")
            return None
        result = self._read_usb("DOA_VALUE_RADIANS")
        if result is None:
            return None
        return float(result[0]), bool(result[1])
