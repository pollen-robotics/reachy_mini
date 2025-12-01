"""Camera constants for Reachy Mini."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List

import numpy as np
import numpy.typing as npt


class CameraResolution(Enum):
    """Base class for camera resolutions."""

    pass


class ArduCamResolution(CameraResolution):
    """Camera resolutions. Arducam_12MP."""

    R2304x1296at30fps = (2304, 1296, 30)
    R4608x2592at10fps = (4608, 2592, 10)
    R1920x1080at30fps = (1920, 1080, 30)
    R1600x1200at30fps = (1600, 1200, 30)
    R1280x720at30fps = (1280, 720, 30)


class RPICameraResolution(CameraResolution):
    """Camera resolutions. Raspberry Pi Camera.

    Camera supports higher resolutions but the h264 encoder won't follow.
    """

    R1920x1080at60fps = (1920, 1080, 60)
    R1920x1080at30fps = (1920, 1080, 30)
    R1280x720at60fps = (1280, 720, 60)
    R1280x720at30fps = (1280, 720, 30)


class MujocoCameraResolution(CameraResolution):
    """Camera resolutions for Mujoco simulated camera."""

    R1280x720at60fps = (1280, 720, 60)


@dataclass
class CameraSpecs:
    """Base camera specifications."""

    available_resolutions: List[CameraResolution] = field(default_factory=list)
    default_resolution: CameraResolution = ArduCamResolution.R1280x720at30fps
    vid = 0
    pid = 0
    K: npt.NDArray[np.float64] = field(default_factory=lambda: np.eye(3))
    D: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros((5,)))


@dataclass
class ArducamSpecs(CameraSpecs):
    """Arducam camera specifications."""

    available_resolutions = [
        ArduCamResolution.R2304x1296at30fps,
        ArduCamResolution.R4608x2592at10fps,
        ArduCamResolution.R1920x1080at30fps,
        ArduCamResolution.R1600x1200at30fps,
        ArduCamResolution.R1280x720at30fps,
    ]
    default_resolution = ArduCamResolution.R1280x720at30fps
    vid = 0x0C45
    pid = 0x636D
    K = np.array([[550.3564, 0.0, 638.0112], [0.0, 549.1653, 364.589], [0.0, 0.0, 1.0]])
    D = np.array([-0.0694, 0.1565, -0.0004, 0.0003, -0.0983])


@dataclass
class ReachyMiniLiteCamSpecs(CameraSpecs):
    """Reachy Mini Lite camera specifications."""

    available_resolutions = [
        RPICameraResolution.R1920x1080at60fps,
        RPICameraResolution.R1280x720at60fps,
    ]
    default_resolution = RPICameraResolution.R1920x1080at60fps
    vid = 0x38FB
    pid = 0x1002
    K = np.array(
        [
            [821.515, 0.0, 962.241],
            [0.0, 820.830, 542.459],
            [0.0, 0.0, 1.0],
        ]
    )

    D = np.array(
        [
            -2.94475669e-02,
            6.00511974e-02,
            3.57813971e-06,
            -2.96459394e-04,
            -3.79243988e-02,
        ]
    )


@dataclass
class ReachyMiniWirelessCamSpecs(ReachyMiniLiteCamSpecs):
    """Reachy Mini Wireless camera specifications."""

    available_resolutions = [
        RPICameraResolution.R1920x1080at30fps,
        RPICameraResolution.R1280x720at30fps,
    ]
    default_resolution = RPICameraResolution.R1920x1080at30fps


@dataclass
class OlderRPiCamSpecs(ReachyMiniLiteCamSpecs):
    """Older Raspberry Pi camera specifications. Keeping for compatibility."""

    vid = 0x1BCF
    pid = 0x28C4


@dataclass
class MujocoCameraSpecs(CameraSpecs):
    """Mujoco simulated camera specifications."""

    available_resolutions = [
        MujocoCameraResolution.R1280x720at60fps,
    ]
    default_resolution = MujocoCameraResolution.R1280x720at60fps
    # ideal camera matrix
    K = np.array(
        [
            [
                MujocoCameraResolution.R1280x720at60fps.value[0],
                0.0,
                MujocoCameraResolution.R1280x720at60fps.value[0] / 2,
            ],
            [
                0.0,
                MujocoCameraResolution.R1280x720at60fps.value[1],
                MujocoCameraResolution.R1280x720at60fps.value[1] / 2,
            ],
            [0.0, 0.0, 1.0],
        ]
    )
    D = np.zeros((5,))  # no distortion
