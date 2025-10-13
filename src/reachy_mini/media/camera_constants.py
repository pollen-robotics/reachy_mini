"""Camera constants for Reachy Mini."""

from enum import Enum


class CameraResolution(Enum):
    """Camera resolutions. Arducam_12MP."""

    R2304x1296 = (2304, 1296)
    R4608x2592 = (4608, 2592)
    R1920x1080 = (1920, 1080)
    R1600x1200 = (1600, 1200)
    R1280x720 = (1280, 720)
