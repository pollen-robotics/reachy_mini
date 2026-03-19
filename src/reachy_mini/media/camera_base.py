"""Abstract base class for camera backends.

Provides the shared resolution / calibration properties and the common
part of ``set_resolution()`` so that ``GStreamerCamera`` and
``GstWebRTCClient`` don't duplicate them.

Subclasses must implement:
- ``open()``, ``read()``, ``close()`` — lifecycle
- ``_apply_resolution()`` — how to handle a resolution change when the
  pipeline is already playing (restart vs error).

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
from reachy_mini.media.camera_utils import scale_intrinsics


class CameraBase(ABC):
    """Abstract camera backend.

    Attributes:
        camera_specs: Camera specifications (resolutions, intrinsics, …).
        resized_K: Intrinsic matrix rescaled to the current resolution.

    """

    def __init__(self, log_level: str = "INFO") -> None:
        """Initialize shared camera attributes."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self._resolution: Optional[CameraResolution] = None
        self.camera_specs: Optional[CameraSpecs] = None
        self.resized_K: Optional[npt.NDArray[np.float64]] = None

    @property
    def resolution(self) -> tuple[int, int]:
        """Current resolution as ``(width, height)`` in pixels.

        Raises:
            RuntimeError: If the resolution has not been set yet.

        """
        if self._resolution is None:
            raise RuntimeError("Camera resolution is not set.")
        return (self._resolution.value[0], self._resolution.value[1])

    @property
    def framerate(self) -> int:
        """Current frame rate in fps.

        Raises:
            RuntimeError: If the resolution has not been set yet.

        """
        if self._resolution is None:
            raise RuntimeError("Camera resolution is not set.")
        return int(self._resolution.value[2])

    @property
    def K(self) -> Optional[npt.NDArray[np.float64]]:
        """Camera intrinsic matrix rescaled to the current resolution.

        Returns the 3x3 matrix or ``None`` if not yet available.

        """
        return self.resized_K

    @property
    def D(self) -> Optional[npt.NDArray[np.float64]]:
        """Distortion coefficients ``[k1, k2, p1, p2, k3]``, or ``None``."""
        if self.camera_specs is not None:
            return self.camera_specs.D
        return None

    def set_resolution(self, resolution: CameraResolution) -> None:
        """Change the camera resolution.

        Updates the appsink caps and rescales the intrinsic matrix.
        Delegates pipeline-specific behaviour to ``_apply_resolution()``.

        Args:
            resolution: Desired resolution from ``CameraResolution``.

        Raises:
            RuntimeError: If camera specs are not set or if the camera is a
                MuJoCo simulated camera (resolution change not supported).
            ValueError: If the resolution is not in the camera's
                ``available_resolutions``.

        """
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
                f"Resolution not supported by the camera. "
                f"Available resolutions are: {self.camera_specs.available_resolutions}"
            )

        # Rescale intrinsic matrix
        original_K = self.camera_specs.K
        original_size: tuple[int, int] = (
            CameraResolution.R3840x2592at30fps.value[0],
            CameraResolution.R3840x2592at30fps.value[1],
        )
        target_size: tuple[int, int] = (resolution.value[0], resolution.value[1])
        crop_scale = resolution.value[3]
        self.resized_K = scale_intrinsics(
            original_K, original_size, target_size, crop_scale
        )

        self._apply_resolution(resolution)

    @abstractmethod
    def _apply_resolution(self, resolution: CameraResolution) -> None:
        """Apply the resolution to the pipeline.

        Called by ``set_resolution()`` after validation and intrinsic
        rescaling.  Subclasses handle the pipeline state differently:

        - ``GStreamerCamera``: close / reopen if PLAYING.
        - ``GstWebRTCClient``: raise ``RuntimeError`` if PLAYING.

        """
        ...

    @abstractmethod
    def open(self) -> None:
        """Start the camera pipeline."""
        ...

    @abstractmethod
    def read(self) -> Optional[npt.NDArray[np.uint8]]:
        """Pull the latest BGR frame.

        Returns:
            A NumPy array of shape ``(height, width, 3)`` or ``None``.

        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Stop the camera pipeline and release resources."""
        ...
