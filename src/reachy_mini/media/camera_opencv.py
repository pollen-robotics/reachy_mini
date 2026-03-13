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
    This module requires the ``opencv`` optional dependency.
    Install with: ``pip install reachy_mini[opencv]``

    This class is typically used internally by the MediaManager when the
    SOUNDDEVICE_OPENCV backend is selected. Direct usage is possible but
    usually not necessary.

Example usage via MediaManager:
    >>> from reachy_mini.media.media_manager import MediaManager, MediaBackend
    >>>
    >>> # Create media manager with OpenCV backend
    >>> media = MediaManager(backend=MediaBackend.SOUNDDEVICE_OPENCV, log_level="INFO")
    >>>
    >>> # Capture frames
    >>> frame = media.get_frame()
    >>> if frame is not None:
    ...     print(f"Captured frame with shape: {frame.shape}")
    >>>
    >>> # Clean up
    >>> media.close()

"""

import platform
from typing import Optional, Tuple, cast

import numpy as np
import numpy.typing as npt

from reachy_mini.media.camera_constants import (
    ArducamSpecs,
    CameraResolution,
    CameraSpecs,
    GenericWebcamSpecs,
    MujocoCameraSpecs,
    OlderRPiCamSpecs,
    ReachyMiniLiteCamSpecs,
)

from .camera_base import CameraBase

try:
    import cv2
    from cv2_enumerate_cameras import enumerate_cameras
except ImportError as e:
    raise ImportError(
        "The 'opencv-python' and 'cv2_enumerate_cameras' packages are required "
        "for the OpenCV camera backend but could not be imported. "
        "Install the optional 'opencv' dependencies with:\n"
        "  pip install reachy_mini[opencv]"
    ) from e


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
        self._resolution: Optional[CameraResolution] = None

    def set_resolution(self, resolution: CameraResolution) -> None:
        """Set the camera resolution."""
        super().set_resolution(resolution)

        self._resolution = resolution
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution.value[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution.value[1])

    def _find_device_by_vid_pid(
        self,
        vid: int = ReachyMiniLiteCamSpecs.vid,
        pid: int = ReachyMiniLiteCamSpecs.pid,
        apiPreference: int = cv2.CAP_ANY,
    ) -> cv2.VideoCapture | None:
        """Find and return a camera with the specified VID and PID.

        Args:
            vid (int): Vendor ID of the camera. Default is ReachyMiniLiteCamSpecs.vid (0x38FB).
            pid (int): Product ID of the camera. Default is ReachyMiniLiteCamSpecs.pid (0x1002).
            apiPreference (int): Preferred API backend for the camera. Default is cv2.CAP_ANY.
                               On Linux, this automatically uses cv2.CAP_V4L2 for better compatibility.

        Returns:
            cv2.VideoCapture | None: A VideoCapture object if the camera with matching
                VID/PID is found and opened successfully, otherwise None.

        Note:
            This function uses the cv2_enumerate_cameras package to enumerate available
            cameras and find one with the specified USB Vendor ID and Product ID.
            This is useful for selecting specific camera models when multiple cameras
            are connected to the system.

            The Arducam camera creates two /dev/videoX devices that enumerate_cameras
            cannot differentiate, so this function tries to open each potential device
            until it finds a working one.

        """
        if platform.system() == "Linux":
            apiPreference = cv2.CAP_V4L2

        selected_cap = None
        for c in enumerate_cameras(apiPreference):
            if c.vid == vid and c.pid == pid:
                # the Arducam camera creates two /dev/videoX devices
                # that enumerate_cameras cannot differentiate
                try:
                    cap = cv2.VideoCapture(c.index, c.backend)
                    if cap.isOpened():
                        selected_cap = cap
                except Exception as e:
                    print(f"Error opening camera {c.index}: {e}")
        return selected_cap

    def _find_camera(
        self,
        apiPreference: int = cv2.CAP_ANY,
        no_cap: bool = False,
    ) -> Tuple[Optional[cv2.VideoCapture], Optional[CameraSpecs]]:
        """Find and return the Reachy Mini camera.

        Looks for the Reachy Mini camera first, then Arducam, then older Raspberry Pi Camera.
        Returns None if no camera is found. Falls back to generic webcam if no specific camera is detected.

        Args:
            apiPreference (int): Preferred API backend for the camera. Default is cv2.CAP_ANY.
                               Options include cv2.CAP_V4L2 (Linux), cv2.CAP_DSHOW (Windows),
                               cv2.CAP_MSMF (Windows), etc.
            no_cap (bool): If True, close the camera after finding it. Useful for testing
                          camera detection without keeping the camera open. Default is False.

        Returns:
            Tuple[Optional[cv2.VideoCapture], Optional[CameraSpecs]]: A tuple containing:
                - cv2.VideoCapture: A VideoCapture object if the camera is found and opened
                  successfully, otherwise None.
                - CameraSpecs: The camera specifications for the detected camera, or None if
                  no camera was found.

        Note:
            This function tries to detect cameras in the following order:
            1. Reachy Mini Lite Camera (preferred)
            2. Older Raspberry Pi Camera
            3. Arducam
            4. Generic Webcam (fallback)

            The function automatically sets the appropriate video codec (MJPG) for
            Reachy Mini and Raspberry Pi cameras to ensure compatibility.

        """
        cap = self._find_device_by_vid_pid(
            ReachyMiniLiteCamSpecs.vid, ReachyMiniLiteCamSpecs.pid, apiPreference
        )
        if cap is not None:
            fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")  # type: ignore
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            if no_cap:
                cap.release()
            return cap, cast(CameraSpecs, ReachyMiniLiteCamSpecs)

        cap = self._find_device_by_vid_pid(
            OlderRPiCamSpecs.vid, OlderRPiCamSpecs.pid, apiPreference
        )
        if cap is not None:
            fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")  # type: ignore
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            if no_cap:
                cap.release()
            return cap, cast(CameraSpecs, OlderRPiCamSpecs)

        cap = self._find_device_by_vid_pid(
            ArducamSpecs.vid, ArducamSpecs.pid, apiPreference
        )
        if cap is not None:
            if no_cap:
                cap.release()
            return cap, cast(CameraSpecs, ArducamSpecs)

        # Fallback: try to open any available webcam (useful for mockup-sim mode on desktop)
        cap = cv2.VideoCapture(0)
        if cap is not None and cap.isOpened():
            if no_cap:
                cap.release()
            return cap, cast(CameraSpecs, GenericWebcamSpecs)

        return None, None

    def open(self, udp_camera: Optional[str] = None) -> None:
        """Open the camera using OpenCV VideoCapture.

        See CameraBase.open() for complete documentation.
        """
        if udp_camera:
            self.cap = cv2.VideoCapture(udp_camera)
            self.camera_specs = cast(CameraSpecs, MujocoCameraSpecs)
            self._resolution = self.camera_specs.default_resolution
        else:
            self.cap, self.camera_specs = self._find_camera()
            if self.cap is None or self.camera_specs is None:
                raise RuntimeError("Camera not found")

            if self._resolution is None:
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
