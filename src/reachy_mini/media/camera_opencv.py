"""OpenCv camera backend.

This module provides an implementation of the CameraBase class using OpenCV.
"""

from typing import Optional, cast

import cv2
import numpy as np
import numpy.typing as npt

from reachy_mini.media.camera_constants import (
    CameraResolution,
    CameraSpecs,
    MujocoCameraSpecs,
)
from reachy_mini.media.camera_utils import find_camera

from .camera_base import CameraBase


class OpenCVCamera(CameraBase):
    """Camera implementation using OpenCV."""

    def __init__(
        self,
        log_level: str = "INFO",
    ) -> None:
        """Initialize the OpenCV camera."""
        super().__init__(log_level=log_level)
        self.cap: Optional[cv2.VideoCapture] = None

    def set_resolution(self, resolution: CameraResolution) -> None:
        """Set the camera resolution."""
        super().set_resolution(resolution)

        self._resolution = resolution
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution.value[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution.value[1])

    def open(self, udp_camera: Optional[str] = None) -> None:
        """Open the camera using OpenCV VideoCapture."""
        if udp_camera:
            self.cap = cv2.VideoCapture(udp_camera)
            self.camera_specs = cast(CameraSpecs, MujocoCameraSpecs)
            self._resolution = self.camera_specs.default_resolution
        else:
            self.cap, self.camera_specs = find_camera()
            if self.cap is None or self.camera_specs is None:
                raise RuntimeError("Camera not found")

            self._resolution = self.camera_specs.default_resolution
            if self._resolution is None:
                raise RuntimeError("Failed to get default camera resolution.")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution.value[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution.value[1])

        self.resized_K = self.camera_specs.K

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

    def read(self) -> Optional[npt.NDArray[np.uint8]]:
        """Read a frame from the camera.

        Returns:
            The frame as a uint8 numpy array, or None if no frame could be read.

        Raises:
            RuntimeError: If the camera is not opened.

        """
        if self.cap is None:
            raise RuntimeError("Camera is not opened.")
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Ensure uint8 dtype
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        return cast(npt.NDArray[np.uint8], frame)

    def close(self) -> None:
        """Release the camera resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
