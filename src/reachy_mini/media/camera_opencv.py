r"""OpenCV camera backend.

This module provides an implementation of the CameraBase class using OpenCV.
It offers cross-platform camera support with automatic camera detection and
configuration for various Reachy Mini camera models.

The OpenCV camera backend features:
- Cross-platform compatibility (Windows, macOS, Linux)
- Automatic camera detection and model identification
- Support for multiple camera models (Reachy Mini Lite, Beta (Arducam), etc.)
- Resolution and frame rate configuration
- Camera calibration parameter access
- Simulation mode support (Mujoco)

Note:
    This class is typically used internally by the MediaManager when the DEFAULT
    backend is selected. Direct usage is possible but usually not necessary.

Example usage via MediaManager:
    >>> from reachy_mini.media.media_manager import MediaManager, MediaBackend
    >>>
    >>> # Create media manager with OpenCV backend (default)
    >>> media = MediaManager(backend=MediaBackend.DEFAULT, log_level="INFO")
    >>>
    >>> # Capture frames
    >>> frame = media.get_frame()
    >>> if frame is not None:
    ...     print(f"Captured frame with shape: {frame.shape}")
    ...     cv2.imshow("Camera", frame)
    ...     cv2.waitKey(1)
    >>>
    >>> # Get camera information
    >>> if media.camera is not None:
    ...     width, height = media.camera.resolution
    ...     fps = media.camera.framerate
    ...     print(f"Camera: {width}x{height}@{fps}fps")
    ...
    ...     # Access calibration information
    ...     K = media.camera.K
    ...     D = media.camera.D
    ...     if K is not None:
    ...         print(f"Camera matrix:\\n{K}")
    ...     if D is not None:
    ...         print(f"Distortion coefficients: {D}")
    >>>
    >>> # Clean up
    >>> media.close()

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
    """Camera implementation using OpenCV.

    This class implements the CameraBase interface using OpenCV, providing
    cross-platform camera support for Reachy Mini robots. It automatically
    detects and configures supported camera models.

    Attributes:
        Inherits all attributes from CameraBase.
        Additionally manages OpenCV VideoCapture objects and camera connections.

    """

    def __init__(
        self,
        log_level: str = "INFO",
    ) -> None:
        """Initialize the OpenCV camera.

        Args:
            log_level (str): Logging level for camera operations.
                          Default: 'INFO'.

        Note:
            This constructor initializes the OpenCV camera system. The actual
            camera device is opened when the open() method is called.

        """
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
        """Open the camera using OpenCV VideoCapture.

        See CameraBase.open() for complete documentation.
        """
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

            self.set_resolution(self._resolution)

            # example of camera controls settings:
            # self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
            # self.cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
            # self.cap.set(cv2.CAP_PROP_SATURATION, 64)

        # self.resized_K = self.camera_specs.K

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

    def read(self) -> Optional[npt.NDArray[np.uint8]]:
        """Read a frame from the camera.

        See CameraBase.read() for complete documentation.

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
        """Release the camera resource.

        See CameraBase.close() for complete documentation.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
