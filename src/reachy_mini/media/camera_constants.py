r"""Camera constants for Reachy Mini.

This module defines camera specifications and resolutions for various camera models
used with the Reachy Mini robot. It includes camera calibration parameters,
supported resolutions, and camera identification information.

The module provides:
- CameraResolution enum: Standardized resolutions and frame rates
- CameraSpecs dataclass: Base camera specifications with calibration data
- Specific camera specifications for different camera models

Example usage:
    >>> from reachy_mini.media.camera_constants import CameraResolution, ReachyMiniLiteCamSpecs
    >>>
    >>> # Get available resolutions for Reachy Mini Lite Camera
    >>> print("Available resolutions:")
    >>> for res in ReachyMiniLiteCamSpecs.available_resolutions:
    ...     width, height, fps = res.value
    ...     print(f"  {width}x{height}@{fps}fps")
    >>>
    >>> # Access camera calibration parameters
    >>> print(f"Camera matrix:\\n{ReachyMiniLiteCamSpecs.K}")
    >>> print(f"Distortion coefficients: {ReachyMiniLiteCamSpecs.D}")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List

import numpy as np
import numpy.typing as npt


class CameraResolution(Enum):
    """Base class for camera resolutions.

    Enumeration of standardized camera resolutions and frame rates supported
    by Reachy Mini cameras. Each enum value contains a tuple of (width, height, fps).

    Attributes:
        R1536x864at40fps: 1536x864 resolution at 40 fps
        R1280x720at60fps: 1280x720 resolution at 60 fps (HD)
        R1280x720at30fps: 1280x720 resolution at 30 fps (HD)
        R1920x1080at30fps: 1920x1080 resolution at 30 fps (Full HD)
        R1920x1080at60fps: 1920x1080 resolution at 60 fps (Full HD)
        R2304x1296at30fps: 2304x1296 resolution at 30 fps
        R1600x1200at30fps: 1600x1200 resolution at 30 fps
        R3264x2448at30fps: 3264x2448 resolution at 30 fps
        R3264x2448at10fps: 3264x2448 resolution at 10 fps
        R3840x2592at30fps: 3840x2592 resolution at 30 fps
        R3840x2592at10fps: 3840x2592 resolution at 10 fps
        R3840x2160at30fps: 3840x2160 resolution at 30 fps (4K UHD)
        R3840x2160at10fps: 3840x2160 resolution at 10 fps (4K UHD)
        R3072x1728at10fps: 3072x1728 resolution at 10 fps
        R4608x2592at10fps: 4608x2592 resolution at 10 fps

    Note:
        The enum values are tuples containing (width, height, frames_per_second).
        Not all resolutions are supported by all camera models - check the specific
        camera specifications for available resolutions.

    Example:
        ```python
        from reachy_mini.media.camera_constants import CameraResolution

        # Get resolution information
        res = CameraResolution.R1280x720at30fps
        width, height, fps = res.value
        print(f"Resolution: {width}x{height}@{fps}fps")

        # Check if a resolution is supported by a camera
        from reachy_mini.media.camera_constants import ReachyMiniLiteCamSpecs
        res = CameraResolution.R1920x1080at60fps
        if res in ReachyMiniLiteCamSpecs.available_resolutions:
            print("This resolution is supported")
        ```

    """

    # TODO check that adding crop factor here doesn't break anything
    # (width, height, fps, crop_factor)
    R1536x864at40fps = (1536, 864, 40, 1.0)

    R1280x720at60fps = (1280, 720, 60, 1.0)
    R1280x720at30fps = (1280, 720, 30, 1.0)

    R1920x1080at30fps = (1920, 1080, 30, 1.115)
    R1920x1080at60fps = (1920, 1080, 60, 1.115)

    R2304x1296at30fps = (2304, 1296, 30, 1.0)
    R1600x1200at30fps = (1600, 1200, 30, 1.0)

    R3264x2448at30fps = (3264, 2448, 30, 1.115)
    R3264x2448at10fps = (3264, 2448, 10, 1.115)

    R3840x2592at30fps = (3840, 2592, 30, 1.0)
    R3840x2592at10fps = (3840, 2592, 10, 1.0)
    R3840x2160at30fps = (3840, 2160, 30, 1.109)
    R3840x2160at10fps = (3840, 2160, 10, 1.109)

    R3072x1728at10fps = (3072, 1728, 10, 1.0)

    R4608x2592at10fps = (4608, 2592, 10, 1.0)


@dataclass
class CameraSpecs:
    """Base camera specifications.

    Dataclass containing specifications for a camera model, including supported
    resolutions, calibration parameters, and USB identification information.

    Attributes:
        name (str): Human-readable name of the camera model.
        available_resolutions (List[CameraResolution]): List of supported resolutions
            and frame rates for this camera model.
        default_resolution (CameraResolution): Default resolution used when the camera
            is initialized.
        vid (int): USB Vendor ID for identifying this camera model.
        pid (int): USB Product ID for identifying this camera model.
        K (npt.NDArray[np.float64]): 3x3 camera intrinsic matrix containing focal
            lengths and principal point coordinates.
        D (npt.NDArray[np.float64]): 5-element array containing distortion coefficients
            (k1, k2, p1, p2, k3) for radial and tangential distortion.

    Note:
        The intrinsic matrix K has the format:
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

        Where fx, fy are focal lengths in pixels, and cx, cy are the principal
        point coordinates (typically near the image center).

    Example:
        ```python
        from reachy_mini.media.camera_constants import CameraSpecs

        # Create a custom camera specification
        custom_specs = CameraSpecs(
            name="custom_camera",
            available_resolutions=[CameraResolution.R1280x720at30fps],
            default_resolution=CameraResolution.R1280x720at30fps,
            vid=0x1234,
            pid=0x5678,
            K=np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]]),
            D=np.zeros(5)
        )
        ```

    """

    name: str = ""
    available_resolutions: List[CameraResolution] = field(default_factory=list)
    default_resolution: CameraResolution = CameraResolution.R1280x720at30fps
    vid = 0
    pid = 0
    K: npt.NDArray[np.float64] = field(default_factory=lambda: np.eye(3))
    D: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros((5,)))


@dataclass
class ArducamSpecs(CameraSpecs):
    """Arducam camera specifications."""

    name = "arducam"
    available_resolutions = [
        CameraResolution.R2304x1296at30fps,
        CameraResolution.R4608x2592at10fps,
        CameraResolution.R1920x1080at30fps,
        CameraResolution.R1600x1200at30fps,
        CameraResolution.R1280x720at30fps,
    ]
    default_resolution = CameraResolution.R1280x720at30fps
    vid = 0x0C45
    pid = 0x636D
    K = np.array([[550.3564, 0.0, 638.0112], [0.0, 549.1653, 364.589], [0.0, 0.0, 1.0]])
    D = np.array([-0.0694, 0.1565, -0.0004, 0.0003, -0.0983])


@dataclass
class ReachyMiniLiteCamSpecs(CameraSpecs):
    """Reachy Mini Lite camera specifications."""

    name = "lite"
    available_resolutions = [
        CameraResolution.R1920x1080at60fps,
        CameraResolution.R3840x2592at30fps,
        CameraResolution.R3840x2160at30fps,
        CameraResolution.R3264x2448at30fps,
    ]
    default_resolution = CameraResolution.R1920x1080at60fps
    vid = 0x38FB
    pid = 0x1002
    # K = np.array(
        # [
            # [821.515, 0.0, 962.241],
            # [0.0, 820.830, 542.459],
            # [0.0, 0.0, 1.0],
        # ]
    # )
    K = np.array(
        [
            [2001.8076426486707, 0.0,  1905.876059826701],
            [0.0, 2003.0778885944105, 1328.3239717935594],
            [0.0, 0.0, 1.0],
        ]
    )

    D = np.array(
        [
          -1.4652320301298614,
          0.6542714131667414,
          0.012147809271745049,
          -0.002677286460143648,
          0.3035939941825349,
          -1.4300809080461876,
          0.570024082887235,
          0.3567299243352951,
          0.003057363348400015,
          0.0003357614008682464,
          -0.009897126394310923,
          -0.002050919484589521,
        ]
    )


@dataclass
class ReachyMiniWirelessCamSpecs(ReachyMiniLiteCamSpecs):
    """Reachy Mini Wireless camera specifications."""

    name = "wireless"
    available_resolutions = [
        CameraResolution.R1280x720at30fps,  # Default for H264 Level 3.1 (Safari/WebKit)
        CameraResolution.R1920x1080at30fps,
        CameraResolution.R1280x720at60fps,
        CameraResolution.R3840x2592at10fps,
        CameraResolution.R3840x2160at10fps,
        CameraResolution.R3264x2448at10fps,
        CameraResolution.R3072x1728at10fps,
    ]
    # 720p@30fps for H264 Level 3.1 compatibility (Safari/WebKit)
    default_resolution = CameraResolution.R1280x720at30fps


@dataclass
class OlderRPiCamSpecs(ReachyMiniLiteCamSpecs):
    """Older Raspberry Pi camera specifications. Keeping for compatibility."""

    name = "older_rpi"
    vid = 0x1BCF
    pid = 0x28C4


@dataclass
class MujocoCameraSpecs(CameraSpecs):
    """Mujoco simulated camera specifications."""

    available_resolutions = [
        CameraResolution.R1280x720at60fps,
    ]
    default_resolution = CameraResolution.R1280x720at60fps
    # ideal camera matrix
    K = np.array(
        [
            [
                CameraResolution.R1280x720at60fps.value[0],
                0.0,
                CameraResolution.R1280x720at60fps.value[0] / 2,
            ],
            [
                0.0,
                CameraResolution.R1280x720at60fps.value[1],
                CameraResolution.R1280x720at60fps.value[1] / 2,
            ],
            [0.0, 0.0, 1.0],
        ]
    )
    D = np.zeros((5,))  # no distortion


@dataclass
class GenericWebcamSpecs(CameraSpecs):
    """Generic webcam specifications (fallback for any webcam)."""

    name = "generic"
    available_resolutions = [
        CameraResolution.R1280x720at30fps,
        CameraResolution.R1920x1080at30fps,
    ]
    default_resolution = CameraResolution.R1280x720at30fps
    # Approximate camera matrix for generic 720p webcam
    K = np.array(
        [
            [640.0, 0.0, 640.0],
            [0.0, 640.0, 360.0],
            [0.0, 0.0, 1.0],
        ]
    )
    D = np.zeros((5,))  # assume no distortion
