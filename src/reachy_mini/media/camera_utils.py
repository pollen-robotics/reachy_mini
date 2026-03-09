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


def undistort_points(
    u: float,
    v: float,
    K: npt.NDArray[np.float64],
    D: npt.NDArray[np.float64],
    max_iterations: int = 20,
    epsilon: float = 0.01,
) -> Tuple[float, float]:
    """Undistort a single pixel coordinate to normalized camera coordinates.

    Pure numpy equivalent of cv2.undistortPoints(). Supports the OpenCV distortion
    model with up to 12 coefficients (rational model + thin prism):
        D = (k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4)

    Also works with 5-coefficient models (k1, k2, p1, p2, k3) and zero-distortion.

    The algorithm matches OpenCV's cvUndistortPointsInternal:
        1. Remove camera intrinsics to get normalized distorted coordinates.
        2. Iteratively solve for undistorted coordinates using a damped
           fixed-point method with adaptive step size.

    Args:
        u: Horizontal pixel coordinate.
        v: Vertical pixel coordinate.
        K: 3x3 camera intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
        D: Distortion coefficients array. Supports lengths 0, 4, 5, 8, 12, or 14.
            Unused positions default to 0.
        max_iterations: Maximum number of iterations (default 20).
        epsilon: Convergence threshold in pixel reprojection error (default 0.01).

    Returns:
        Tuple (x_n, y_n): Normalized undistorted coordinates (on the z=1 plane).

    Reference:
        OpenCV distortion model and undistortPoints algorithm:
        https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
        https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/undistort.dispatch.cpp

    """
    # Extract intrinsics
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    # Step 1: Remove intrinsics to get normalized distorted coordinates
    x0 = (u - cx) / fx
    y0 = (v - cy) / fy

    # Pad D to 14 elements so indexing is safe (OpenCV convention)
    d = np.zeros(14)
    n = min(len(D), 14)
    d[:n] = D[:n]

    # OpenCV coefficient ordering: k1=d[0], k2=d[1], p1=d[2], p2=d[3], k3=d[4],
    # k4=d[5], k5=d[6], k6=d[7], s1=d[8], s2=d[9], s3=d[10], s4=d[11]

    # Step 2: Damped fixed-point iteration matching OpenCV's algorithm.
    # We want to find (x, y) such that distort(x, y) = (x0, y0).
    x = x0
    y = y0
    alpha = 1.0  # damping factor
    prev_error = float("inf")

    for _ in range(max_iterations):
        r2 = x * x + y * y

        # icdist = (1 + k4*r2 + k5*r4 + k6*r6) / (1 + k1*r2 + k2*r4 + k3*r6)
        # This is the inverse of the radial distortion factor.
        numerator = 1.0 + (d[7] * r2 + d[6]) * r2 + d[5]  # k6*r2 + k5
        numerator = numerator * r2 + 1.0  # full: 1 + k4*r2 + k5*r4 + k6*r6
        # Recompute correctly using Horner's method:
        numerator = 1.0 + ((d[7] * r2 + d[6]) * r2 + d[5]) * r2
        denominator = 1.0 + ((d[4] * r2 + d[1]) * r2 + d[0]) * r2

        if denominator == 0.0:
            icdist = 1.0
        else:
            icdist = numerator / denominator

        if icdist < 0:
            # Distortion model is invalid at this radius, fall back to pinhole
            return float(x0), float(y0)

        # Tangential distortion
        delta_x = (
            2.0 * d[2] * x * y + d[3] * (r2 + 2.0 * x * x) + d[8] * r2 + d[9] * r2 * r2
        )
        delta_y = (
            d[2] * (r2 + 2.0 * y * y)
            + 2.0 * d[3] * x * y
            + d[10] * r2
            + d[11] * r2 * r2
        )

        # Damped fixed-point update
        new_x = (1.0 - alpha) * x + alpha * (x0 - delta_x) * icdist
        new_y = (1.0 - alpha) * y + alpha * (y0 - delta_y) * icdist

        # Compute reprojection error to check convergence
        # Forward-project (new_x, new_y) back to pixel coordinates
        nr2 = new_x * new_x + new_y * new_y
        nr4 = nr2 * nr2
        nr6 = nr4 * nr2
        cdist = 1.0 + d[0] * nr2 + d[1] * nr4 + d[4] * nr6
        icdist2_den = 1.0 + d[5] * nr2 + d[6] * nr4 + d[7] * nr6
        icdist2 = 1.0 / icdist2_den if icdist2_den != 0.0 else 1.0

        a1 = 2.0 * new_x * new_y
        a2 = nr2 + 2.0 * new_x * new_x
        a3 = nr2 + 2.0 * new_y * new_y

        xd = new_x * cdist * icdist2 + d[2] * a1 + d[3] * a2 + d[8] * nr2 + d[9] * nr4
        yd = new_y * cdist * icdist2 + d[2] * a3 + d[3] * a1 + d[10] * nr2 + d[11] * nr4

        x_proj = xd * fx + cx
        y_proj = yd * fy + cy
        error = ((x_proj - u) ** 2 + (y_proj - v) ** 2) ** 0.5

        if error < epsilon:
            return float(new_x), float(new_y)

        if error > prev_error:
            # Reduce step size when diverging
            alpha *= 0.5
        else:
            x = new_x
            y = new_y

        prev_error = error

    return float(x), float(y)


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
