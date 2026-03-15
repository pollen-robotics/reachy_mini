"""Camera geometry utilities for Reachy Mini.

This module provides pure NumPy implementations of camera geometry functions
for Reachy Mini robots. These functions handle camera calibration mathematics
including point undistortion and intrinsic matrix scaling without requiring
OpenCV or other computer vision libraries.

Note:
    This module does NOT require OpenCV and contains only pure mathematical
    operations. For camera frame reading, see
    ``reachy_mini.media.camera_gstreamer`` (local IPC reader) or
    ``reachy_mini.media.webrtc_client_gstreamer`` (WebRTC client).

Functions:
    undistort_points: Convert distorted pixel coordinates to normalized camera coordinates.
    scale_intrinsics: Scale camera intrinsics for different resolutions with cropping.

Example:
    ```python
    from reachy_mini.media.camera_utils import undistort_points, scale_intrinsics
    import numpy as np

    # Undistort a pixel coordinate using camera intrinsics K and distortion D
    K = np.array([[800.0, 0, 640.0], [0, 600.0, 360.0], [0, 0, 1.0]])
    D = np.zeros(5)  # No distortion
    x_n, y_n = undistort_points(800.0, 480.0, K, D)

    # Scale intrinsics for a different resolution with cropping
    K_original = np.array([[1000.0, 0, 640.0], [0, 1000.0, 360.0], [0, 0, 1.0]])
    K_new = scale_intrinsics(
        K_original,
        original_size=(1280, 720),
        target_size=(640, 480),
        crop_scale=1.0
    )
    ```

"""

from typing import Tuple

import numpy as np
import numpy.typing as npt


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
