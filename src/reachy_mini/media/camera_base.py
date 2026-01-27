"""Base classes for camera implementations.

The camera implementations support various backends and provide a unified
interface for capturing images. This module defines the abstract base class
that all camera implementations should inherit from, ensuring consistent
API across different camera backends.

Available backends include:
- OpenCV: Cross-platform camera backend using OpenCV library
- GStreamer: GStreamer-based camera backend for advanced video processing
- WebRTC: WebRTC-based camera for real-time video communication

Example usage:
    >>> from reachy_mini.media.camera_base import CameraBase
    >>> class MyCamera(CameraBase):
    ...     def open(self) -> None:
    ...         pass
    ...     def read(self) -> Optional[npt.NDArray[np.uint8]]:
    ...         pass
    ...     def close(self) -> None:
    ...         pass
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
    """Abstract class for opening and managing a camera.

    This class defines the interface that all camera implementations must follow.
    It provides common camera parameters and methods for managing camera devices,
    including image capture, resolution management, and camera calibration.

    Attributes:
        logger (logging.Logger): Logger instance for camera-related messages.
        _resolution (Optional[CameraResolution]): Current camera resolution setting.
        camera_specs (Optional[CameraSpecs]): Camera specifications including
            supported resolutions and calibration parameters.
        resized_K (Optional[npt.NDArray[np.float64]]): Camera intrinsic matrix
            resized to match the current resolution.

    """

    def __init__(
        self,
        log_level: str = "INFO",
    ) -> None:
        """Initialize the camera.

        Args:
            log_level (str): Logging level for camera operations.
                          Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
                          Default: 'INFO'.

        Note:
            This constructor initializes the logging system. Camera specifications
            and resolution should be set before calling open().

        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self._resolution: Optional[CameraResolution] = None
        self.camera_specs: Optional[CameraSpecs] = None
        self.resized_K: Optional[npt.NDArray[np.float64]] = None

    @property
    def resolution(self) -> tuple[int, int]:
        """Get the current camera resolution as a tuple (width, height).

        Returns:
            tuple[int, int]: A tuple containing (width, height) in pixels.

        Raises:
            RuntimeError: If camera resolution has not been set.

        Example:
            ```python
            width, height = camera.resolution
            print(f"Camera resolution: {width}x{height}")
            ```

        """
        if self._resolution is None:
            raise RuntimeError("Camera resolution is not set.")
        return (self._resolution.value[0], self._resolution.value[1])

    @property
    def framerate(self) -> int:
        """Get the current camera frames per second.

        Returns:
            int: The current frame rate in frames per second (fps).

        Raises:
            RuntimeError: If camera resolution has not been set.

        Example:
            ```python
            fps = camera.framerate
            print(f"Camera frame rate: {fps} fps")
            ```

        """
        if self._resolution is None:
            raise RuntimeError("Camera resolution is not set.")
        return int(self._resolution.value[2])

    @property
    def K(self) -> Optional[npt.NDArray[np.float64]]:
        """Get the camera intrinsic matrix for the current resolution.

        Returns:
            Optional[npt.NDArray[np.float64]]: The 3x3 camera intrinsic matrix
            in the format:

            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]

            Where fx, fy are focal lengths in pixels, and cx, cy are the
            principal point coordinates. Returns None if not available.

        Note:
            The intrinsic matrix is automatically resized to match the current
            camera resolution when set_resolution() is called.

        Example:
            ```python
            K = camera.K
            if K is not None:
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
            ```

        """
        return self.resized_K

    @property
    def D(self) -> Optional[npt.NDArray[np.float64]]:
        """Get the camera distortion coefficients.

        Returns:
            Optional[npt.NDArray[np.float64]]: The distortion coefficients
            as a 5-element array [k1, k2, p1, p2, k3] representing radial
            and tangential distortion parameters, or None if not available.

        Note:
            These coefficients can be used with OpenCV's distortion correction
            functions to undistort captured images.

        Example:
            ```python
            D = camera.D
            if D is not None:
                print(f"Distortion coefficients: {D}")
            ```

        """
        if self.camera_specs is not None:
            return self.camera_specs.D
        return None

    def set_resolution(self, resolution: CameraResolution) -> None:
        """Set the camera resolution.

        Args:
            resolution (CameraResolution): The desired camera resolution from
                the CameraResolution enum.

        Raises:
            RuntimeError: If camera specs are not set or if trying to change
                        resolution of a Mujoco simulated camera.
            ValueError: If the requested resolution is not supported by the camera.

        Note:
            This method updates the camera's resolution and automatically rescales
            the camera intrinsic matrix (K) to match the new resolution. The
            rescaling preserves the camera's field of view and principal point
            position relative to the image dimensions.

        Example:
            ```python
            from reachy_mini.media.camera_constants import CameraResolution
            camera.set_resolution(CameraResolution.R1280x720at30fps)
            ```

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
                f"Resolution not supported by the camera. Available resolutions are : {self.camera_specs.available_resolutions}"
            )

        original_K = self.camera_specs.K
        original_size: tuple[int, int] = (CameraResolution.R3840x2592at30fps.value[0], CameraResolution.R3840x2592at30fps.value[1])
        target_size: tuple[int, int] = (resolution.value[0], resolution.value[1])
        crop_scale = resolution.value[3]

        self.resized_K = scale_intrinsics(
            original_K, original_size, target_size, crop_scale
        )

    @abstractmethod
    def open(self) -> None:
        """Open the camera.

        This method should initialize the camera device and prepare it for
        capturing images. After calling this method, read() should be able
        to retrieve camera frames.

        Note:
            Implementations should handle any necessary resource allocation,
            camera configuration, and error checking. If the camera cannot
            be opened, implementations should log appropriate error messages.

        Raises:
            RuntimeError: If the camera cannot be opened due to hardware
                        or configuration issues.

        """
        pass

    @abstractmethod
    def read(self) -> Optional[npt.NDArray[np.uint8]]:
        """Read an image from the camera. Returns the image or None if error.

        Returns:
            Optional[npt.NDArray[np.uint8]]: A numpy array containing the
            captured image in BGR format (OpenCV convention), or None if
            no image is available or an error occurred.

            The array shape is (height, width, 3) where the last dimension
            represents the BGR color channels.

        Note:
            This method should be called after open() has been called.
            The image resolution can be obtained via the resolution property.

        Example:
            ```python
            camera.open()
            frame = camera.read()
            if frame is not None:
                cv2.imshow("Camera Frame", frame)
                cv2.waitKey(1)
            ```

        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the camera and release resources.

        This method should stop any ongoing image capture and release
        all associated resources. After calling this method, read() should
        return None until open() is called again.

        Note:
            Implementations should ensure proper cleanup to prevent resource leaks.

        """
        pass
