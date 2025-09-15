"""Base classes for audio implementations.

The audio implementations support various backends and provide a unified
interface for audio input/output.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum


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
    def open(self):
        """Open the audio device."""
        pass

    @abstractmethod
    def read(self):
        """Read audio data from the device. Returns the data or None if error."""
        pass

    @abstractmethod
    def close(self):
        """Close the audio device and release resources."""
        pass
