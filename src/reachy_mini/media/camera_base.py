"""Base classes for camera implementations.

The camera implementations support various backends and provide a unified
interface for capturing images.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import numpy.typing as npt

from reachy_mini.media.camera_constants import (
    CameraResolution,
    CameraSpecs,
    MujocoCameraSpecs,
)


class CameraBase(ABC):
    """Abstract class for opening and managing a camera."""

    def __init__(
        self,
        log_level: str = "INFO",
    ) -> None:
        """Initialize the camera."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self._resolution: Optional[CameraResolution] = None
        self.camera_specs: Optional[CameraSpecs] = None
        self.resized_K: Optional[npt.NDArray[np.float64]] = None

    @property
    def resolution(self) -> tuple[int, int]:
        """Get the current camera resolution as a tuple (width, height)."""
        if self._resolution is None:
            raise RuntimeError("Camera resolution is not set.")
        return (self._resolution.value[0], self._resolution.value[1])

    @property
    def framerate(self) -> int:
        """Get the current camera frames per second."""
        if self._resolution is None:
            raise RuntimeError("Camera resolution is not set.")
        return int(self._resolution.value[2])

    @property
    def K(self) -> Optional[npt.NDArray[np.float64]]:
        """Get the camera intrinsic matrix for the current resolution."""
        return self.resized_K

    @property
    def D(self) -> Optional[npt.NDArray[np.float64]]:
        """Get the camera distortion coefficients."""
        if self.camera_specs is not None:
            return self.camera_specs.D
        return None

    def set_resolution(self, resolution: CameraResolution) -> None:
        """Set the camera resolution."""
        if self.camera_specs is None:
            raise RuntimeError(
                "Camera specs not set. Open the camera before setting the resolution."
            )

        if isinstance(self.camera_specs, MujocoCameraSpecs):
            raise RuntimeError(
                "Cannot change resolution of Mujoco simulated camera for now."
            )

        if resolution not in self.camera_specs.available_resolutions:
            raise ValueError(
                f"Resolution not supported by the camera. Available resolutions are : {self.camera_specs.available_resolutions}"
            )

        w_ratio = resolution.value[0] / self.camera_specs.default_resolution.value[0]
        h_ratio = resolution.value[1] / self.camera_specs.default_resolution.value[1]
        self.resized_K = self.camera_specs.K.copy()

        self.resized_K[0, 0] *= w_ratio
        self.resized_K[1, 1] *= h_ratio
        self.resized_K[0, 2] *= w_ratio
        self.resized_K[1, 2] *= h_ratio

    @abstractmethod
    def open(self) -> None:
        """Open the camera."""
        pass

    @abstractmethod
    def read(self) -> Optional[npt.NDArray[np.uint8]]:
        """Read an image from the camera. Returns the image or None if error."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the camera and release resources."""
        pass
