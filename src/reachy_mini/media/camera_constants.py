"""Camera constants for Reachy Mini."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List

import numpy as np
import numpy.typing as npt


class CameraResolution(Enum):
    """Base class for camera resolutions."""

    R1536x864at40fps = (1536, 864, 40)

    R1280x720at60fps = (1280, 720, 60)
    R1280x720at30fps = (1280, 720, 30)

    R1920x1080at30fps = (1920, 1080, 30)
    R1920x1080at60fps = (1920, 1080, 60)

    R2304x1296at30fps = (2304, 1296, 30)
    R1600x1200at30fps = (1600, 1200, 30)

    R3264x2448at30fps = (3264, 2448, 30)
    R3264x2448at10fps = (3264, 2448, 10)

    R3840x2592at30fps = (3840, 2592, 30)
    R3840x2592at10fps = (3840, 2592, 10)
    R3840x2160at30fps = (3840, 2160, 30)
    R3840x2160at10fps = (3840, 2160, 10)

    R3072x1728at10fps = (3072, 1728, 10)

    R4608x2592at10fps = (4608, 2592, 10)


@dataclass
class CameraSpecs:
    """Base camera specifications."""

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

    name = "wireless"
    available_resolutions = [
        CameraResolution.R1920x1080at30fps,
        CameraResolution.R1280x720at60fps,
        CameraResolution.R3840x2592at10fps,
        CameraResolution.R3840x2160at10fps,
        CameraResolution.R3264x2448at10fps,
        CameraResolution.R3072x1728at10fps,
    ]
    default_resolution = CameraResolution.R1920x1080at30fps


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
