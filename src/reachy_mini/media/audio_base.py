"""Base classes for audio implementations.

The audio implementations support various backends and provide a unified
interface for audio input/output.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import numpy.typing as npt

from reachy_mini.media.audio_control_utils import ReSpeaker, init_respeaker_usb


class AudioBase(ABC):
    """Abstract class for opening and managing audio devices."""

    SAMPLE_RATE = 16000  # respeaker samplerate
    CHANNELS = 2  # respeaker channels

    def __init__(self, log_level: str = "INFO") -> None:
        """Initialize the audio device."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self._respeaker: Optional[ReSpeaker] = init_respeaker_usb()

    def __del__(self) -> None:
        """Destructor to ensure resources are released."""
        if self._respeaker:
            self._respeaker.close()

    @abstractmethod
    def start_recording(self) -> None:
        """Start recording audio."""
        pass

    @abstractmethod
    def get_audio_sample(self) -> Optional[npt.NDArray[np.float32]]:
        """Read audio data from the device. Returns the data or None if error."""
        pass

    def get_input_audio_samplerate(self) -> int:
        """Get the input samplerate of the audio device."""
        return self.SAMPLE_RATE

    def get_output_audio_samplerate(self) -> int:
        """Get the outputsamplerate of the audio device."""
        return self.SAMPLE_RATE

    def get_input_channels(self) -> int:
        """Get the number of input channels of the audio device."""
        return self.CHANNELS

    def get_output_channels(self) -> int:
        """Get the number of output channels of the audio device."""
        return self.CHANNELS

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

    def get_DoA(self) -> tuple[float, bool] | None:
        """Get the Direction of Arrival (DoA) value from the ReSpeaker device.

        The spatial angle is given in radians:
        0 radians is left, π/2 radians is front/back, π radians is right.

        Note: The microphone array requires firmware version 2.1.0 or higher to support this feature.
        The firmware is located in src/reachy_mini/assets/firmware/*.bin.
        Refer to https://wiki.seeedstudio.com/respeaker_xvf3800_introduction/#update-firmware for the upgrade process.

        Returns:
            tuple: A tuple containing the DoA value as a float (radians) and the speech detection as a bool, or None if the device is not found.

        """
        if not self._respeaker:
            self.logger.warning("ReSpeaker device not found.")
            return None

        result = self._respeaker.read("DOA_VALUE_RADIANS")
        if result is None:
            return None
        return float(result[0]), bool(result[1])
