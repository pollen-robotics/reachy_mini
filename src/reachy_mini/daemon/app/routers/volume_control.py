import logging
import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

SOUND_CARD_NAMES = ["Reachy Mini Audio", "respeaker"]

class DeviceType(Enum):
    INPUT = "input"
    OUTPUT = "output"

@dataclass
class VolumeControl(ABC):
    """
    Base class for volume control.
    """
    
    logger: logging.Logger = field(init=False, default_factory=lambda: logging.getLogger(f"[VolumeControl {platform.system()}]"))

    @abstractmethod
    def set_output_volume(volume: float) -> bool:
        """Set the output volume to the provided value between 0 (minmum volume) and 1 (maximum volume)"""
        pass

    @abstractmethod
    def get_output_volume() -> float:
        """Get the output volume as a value between 0 (minimum volume) and 1 (maximum volume)"""
        pass

    @abstractmethod
    def set_input_volume(volume: float) -> bool:
        """Set the input volume to the provided value between 0 (minmum volume) and 1 (maximum volume)"""
        pass

    @abstractmethod
    def get_input_volume() -> float:
        """Get the input volume as a value between 0 (minimum volume) and 1 (maximum volume)"""
        pass


def create_volume_control() -> VolumeControl:
    """Factory that returns the correct VolumeControl subclass for the current platform.
    
    Imports are lazy to avoid loading platform-specific dependencies on the wrong OS
    (e.g. CoreAudio on Linux, pycaw on macOS).
    
    Returns:
        A VolumeControl instance for the current platform.
    
    Raises:
        RuntimeError: If the current platform is not supported.
    """
    system = platform.system()

    if system == "Darwin":
        from .volume_control_macos import VolumeControlMacOS
        return VolumeControlMacOS()
    elif system == "Linux":
        raise RuntimeError(f"Linux volume control is not implemented yet")
    elif system == "Windows":
        raise RuntimeError(f"Windows volume control is not implemented yet")
    else:
        raise RuntimeError(f"Unsupported platform for volume control: {system}")