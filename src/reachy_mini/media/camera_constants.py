"""Camera constants for Reachy Mini."""

from enum import Enum


class CameraResolution(Enum):
    """Camera resolutions. Arducam_12MP."""

    R2304x1296 = (2304, 1296, 30)
    R4608x2592 = (4608, 2592, 10)
    R1920x1080 = (1920, 1080, 30)
    R1600x1200 = (1600, 1200, 30)
    R1280x720 = (1280, 720, 30)


class RPICameraResolution(Enum):
    """Camera resolutions. Raspberry Pi Camera.

    Camera supports higher resolutions but the h264 encoder won't follow.
    """

    R1920x1080 = (1920, 1080, 30)
    R1600x1200 = (1600, 1200, 30)
    R1536x864 = (1536, 864, 40)
    R1280x720 = (1280, 720, 60)
