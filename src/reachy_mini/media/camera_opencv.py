"""OpenCv camera backend.

This module provides an implementation of the CameraBase class using OpenCV.
"""

from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt

from reachy_mini.media.camera_utils import find_camera

from .camera_base import CameraBackend, CameraBase


class OpenCVCamera(CameraBase):
    """Camera implementation using OpenCV."""

    def __init__(self, log_level: str = "INFO") -> None:
        """Initialize the OpenCV camera."""
        super().__init__(backend=CameraBackend.OPENCV, log_level=log_level)
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self, udp_camera: Optional[str] = None) -> None:
        """Open the camera using OpenCV VideoCapture."""
        if udp_camera:
            self.cap = cv2.VideoCapture(udp_camera)
        else:
            self.cap = find_camera()

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

    def read(self) -> Optional[bytes | npt.NDArray[np.uint8]]:
        """Read a frame from the camera. Returns the frame or None if error."""
        if self.cap is None:
            raise RuntimeError("Camera is not opened.")
        ret, frame = self.cap.read()
        if not ret:
            return None

        return np.asarray(frame, dtype=np.uint8)

    def close(self) -> None:
        """Release the camera resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
