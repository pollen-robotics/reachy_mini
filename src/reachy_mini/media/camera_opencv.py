"""OpenCv camera backend.

This module provides an implementation of the CameraBase class using OpenCV.
"""

import cv2

from reachy_mini.media.camera_constants import CameraResolution
from reachy_mini.media.camera_utils import find_camera

from .camera_base import CameraBackend, CameraBase


class OpenCVCamera(CameraBase):
    """Camera implementation using OpenCV."""

    def __init__(
        self,
        log_level: str = "INFO",
        resolution: CameraResolution = CameraResolution.R1280x720,
    ) -> None:
        """Initialize the OpenCV camera."""
        super().__init__(
            backend=CameraBackend.OPENCV, log_level=log_level, resolution=resolution
        )
        self.cap = None

    def open(
        self,
        udp_camera: str = None,
    ):
        """Open the camera using OpenCV VideoCapture."""
        if udp_camera:
            self.cap = cv2.VideoCapture(udp_camera)
        else:
            self.cap = find_camera()
            if self.cap is None:
                raise RuntimeError("Camera not found")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

    def read(self):
        """Read a frame from the camera. Returns the frame or None if error."""
        if self.cap is None:
            raise RuntimeError("Camera is not opened.")
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def close(self):
        """Release the camera resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
