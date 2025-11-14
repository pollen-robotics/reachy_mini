"""Camera constants for Reachy Mini."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List

import numpy as np


class CameraResolution(Enum):
    """Base class for camera resolutions."""

    pass


class ArduCamResolution(CameraResolution):
    """Camera resolutions. Arducam_12MP."""

    R2304x1296 = (2304, 1296, 30)
    R4608x2592 = (4608, 2592, 10)
    R1920x1080 = (1920, 1080, 30)
    R1600x1200 = (1600, 1200, 30)
    R1280x720 = (1280, 720, 30)


class RPICameraResolution(CameraResolution):
    """Camera resolutions. Raspberry Pi Camera.

    Camera supports higher resolutions but the h264 encoder won't follow.
    """

    R1920x1080 = (1920, 1080, 30)
    R1600x1200 = (1600, 1200, 30)
    R1536x864 = (1536, 864, 40)
    R1280x720 = (1280, 720, 60)


@dataclass
class CameraSpecs:
    """Base camera specifications."""

    available_resolutions: List[CameraResolution] = field(default_factory=list)
    default_resolution: CameraResolution = None
    vid = None
    pid = None
    # TODO TEMPORARY
    K = np.array(
        [[550.3564, 0.0, 638.0112], [0.0, 549.1653, 364.589], [0.0, 0.0, 1.0]]
    )  # FOR 1280x720
    D = np.array([-0.0694, 0.1565, -0.0004, 0.0003, -0.0983])


@dataclass
class ArducamSpecs(CameraSpecs):
    """Arducam camera specifications."""

    available_resolutions = [
        ArduCamResolution.R2304x1296,
        ArduCamResolution.R4608x2592,
        ArduCamResolution.R1920x1080,
        ArduCamResolution.R1600x1200,
        ArduCamResolution.R1280x720,
    ]
    default_resolution = ArduCamResolution.R1280x720
    vid = 0x0C45
    pid = 0x636D
    # TODO handle calibration depending on resolution ? How ?
    K = np.array(
        [[550.3564, 0.0, 638.0112], [0.0, 549.1653, 364.589], [0.0, 0.0, 1.0]]
    )  # FOR 1280x720
    D = np.array([-0.0694, 0.1565, -0.0004, 0.0003, -0.0983])


@dataclass
class ReachyMiniCamSpecs(CameraSpecs):
    """Reachy Mini camera specifications."""

    available_resolutions = [
        RPICameraResolution.R1920x1080,
        RPICameraResolution.R1600x1200,
        RPICameraResolution.R1536x864,
        RPICameraResolution.R1280x720,
    ]
    default_resolution = RPICameraResolution.R1920x1080
    vid = 0x38FB
    pid = 0x1002


@dataclass
class OlderRPiCamSpecs(CameraSpecs):
    """Older Raspberry Pi camera specifications."""

    available_resolutions = [
        RPICameraResolution.R1920x1080,
        RPICameraResolution.R1600x1200,
        RPICameraResolution.R1536x864,
        RPICameraResolution.R1280x720,
    ]
    default_resolution = RPICameraResolution.R1920x1080
    vid = 0x1BCF
    pid = 0x28C4
