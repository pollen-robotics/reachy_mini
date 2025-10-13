"""Base classes for camera implementations.

The camera implementations support various backends and provide a unified
interface for capturing images.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import numpy as np
import numpy.typing as npt

from reachy_mini.media.camera_constants import CameraResolution


class CameraBackend(Enum):
    """Camera backends."""

    OPENCV = "opencv"
    GSTREAMER = "gstreamer"


class CameraBase(ABC):
    """Abstract class for opening and managing a camera."""

    def __init__(
        self,
        backend: CameraBackend,
        log_level: str = "INFO",
        resolution: CameraResolution = CameraResolution.R1280x720,
    ) -> None:
        """Initialize the camera."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.backend = backend
        self._resolution = resolution

    @property
    def resolution(self) -> tuple[int, int]:
        """Get the current camera resolution as a tuple (width, height)."""
        return (self._resolution.value[0], self._resolution.value[1])

    @abstractmethod
    def open(self) -> None:
        """Open the camera."""
        pass

    @abstractmethod
    def read(self) -> Optional[bytes | npt.NDArray[np.uint8]]:
        """Read an image from the camera. Returns the image or None if error."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the camera and release resources."""
        pass
