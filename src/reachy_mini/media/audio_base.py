"""Base classes for audio implementations.

The audio implementations support various backends and provide a unified
interface for audio input/output.
"""

import logging
from abc import ABC, abstractmethod


class AudioBase(ABC):
    """Abstract class for opening and managing audio devices."""

    def __init__(self, log_level: str = "INFO") -> None:
        """Initialize the audio device."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

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
