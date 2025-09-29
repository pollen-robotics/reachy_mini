"""Base classes for audio implementations.

The audio implementations support various backends and provide a unified
interface for audio input/output.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import numpy as np
import numpy.typing as npt


class AudioBackend(Enum):
    """Audio backends."""

    SOUNDDEVICE = "sounddevice"
    GSTREAMER = "gstreamer"


class AudioBase(ABC):
    """Abstract class for opening and managing audio devices."""

    def __init__(self, backend: AudioBackend, log_level: str = "INFO") -> None:
        """Initialize the audio device."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.backend = backend

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
