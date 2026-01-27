"""Camera utility for Reachy Mini.

This module provides utility functions for working with cameras on the Reachy Mini robot.
It includes functions for detecting and identifying different camera models, managing
camera connections, and handling camera-specific configurations.

Supported camera types:
- Reachy Mini Lite Camera
- Arducam
- Older Raspberry Pi Camera
- Generic Webcams (fallback)

Example usage:
    >>> from reachy_mini.media.camera_utils import find_camera
    >>>
    >>> # Find and open the Reachy Mini camera
    >>> cap, camera_specs = find_camera()
    >>> if cap is not None:
    ...     print(f"Found {camera_specs.name} camera")
    ...     # Use the camera
    ...     ret, frame = cap.read()
    ...     cap.release()
    ... else:
    ...     print("No camera found")
"""

import platform
from typing import Optional, Tuple, cast

import cv2
import numpy as np
import numpy.typing as npt
from cv2_enumerate_cameras import enumerate_cameras

from reachy_mini.media.camera_constants import (
    ArducamSpecs,
    CameraSpecs,
    GenericWebcamSpecs,
    OlderRPiCamSpecs,
    ReachyMiniLiteCamSpecs,
)


def find_camera(
    apiPreference: int = cv2.CAP_ANY, no_cap: bool = False
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

    Example:
        ```python
        cap, specs = find_camera()
        if cap is not None:
            print(f"Found {specs.name} camera")
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            # Capture a frame
            ret, frame = cap.read()
            cap.release()
        else:
            print("No camera found")
        ```

    """
    cap = find_camera_by_vid_pid(
        ReachyMiniLiteCamSpecs.vid, ReachyMiniLiteCamSpecs.pid, apiPreference
    )
    if cap is not None:
        fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")  # type: ignore
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        if no_cap:
            cap.release()
        return cap, cast(CameraSpecs, ReachyMiniLiteCamSpecs)

    cap = find_camera_by_vid_pid(
        OlderRPiCamSpecs.vid, OlderRPiCamSpecs.pid, apiPreference
    )
    if cap is not None:
        fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")  # type: ignore
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        if no_cap:
            cap.release()
        return cap, cast(CameraSpecs, OlderRPiCamSpecs)

    cap = find_camera_by_vid_pid(ArducamSpecs.vid, ArducamSpecs.pid, apiPreference)
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


def find_camera_by_vid_pid(
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

    Example:
        ```python
        # Find Reachy Mini Lite Camera by its default VID/PID
        cap = find_camera_by_vid_pid()
        if cap is not None:
            print("Found Reachy Mini Lite Camera")
            cap.release()

        # Find a specific camera by custom VID/PID
        cap = find_camera_by_vid_pid(vid=0x0C45, pid=0x636D)  # Arducam
        if cap is not None:
            print("Found Arducam")
        ```
        ...     cap.release()

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


def scale_intrinsics(
    K_original: npt.NDArray[np.float64],
    original_size: Tuple[int, int],
    target_size: Tuple[int, int],
    crop_scale: float,
) -> npt.NDArray[np.float64]:
    """Scale camera intrinsics for a different resolution with cropping.

    Args:
        K_original: Original 3x3 camera matrix
        original_size: (width, height) of original calibration
        target_size: (width, height) of target resolution
        crop_scale: Scale factor due to digital zoom/crop (>1 means more zoomed in)

    Returns:
        K_scaled: Adjusted camera matrix for target resolution

    """
    K_scaled: npt.NDArray[np.float64] = K_original.copy()

    orig_w, orig_h = original_size
    target_w, target_h = target_size

    # Extract original parameters
    fx = K_original[0, 0]
    fy = K_original[1, 1]
    cx = K_original[0, 2]
    cy = K_original[1, 2]

    # Focal length scaling has two components:
    # 1. Resolution scaling: focal length in pixels scales with image dimensions
    # 2. Crop/zoom scaling: cropping increases effective focal length

    resolution_scale_x = target_w / orig_w
    resolution_scale_y = target_h / orig_h

    fx_scaled = fx * resolution_scale_x * crop_scale
    fy_scaled = fy * resolution_scale_y * crop_scale

    # Principal point scales with resolution
    # For centered crop, it stays at the image center after scaling
    cx_scaled = (cx / orig_w) * target_w
    cy_scaled = (cy / orig_h) * target_h

    K_scaled[0, 0] = fx_scaled
    K_scaled[1, 1] = fy_scaled
    K_scaled[0, 2] = cx_scaled
    K_scaled[1, 2] = cy_scaled

    return K_scaled
