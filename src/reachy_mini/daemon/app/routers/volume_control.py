"""Volume control base class and factory for platform-specific implementations."""

import logging
import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

SOUND_CARD_NAMES = ["reachy mini audio", "respeaker"]


class AudioDeviceType(Enum):
    """Type of device: INPUT or OUTPUT."""

    INPUT = "input"
    OUTPUT = "output"


class AudioDevice(NamedTuple):
    """An audio device with its ID, name, and type."""

    id: int | str | None
    name: str
    device_type: AudioDeviceType


@dataclass
class VolumeControl(ABC):
    """Base class for volume control."""

    logger: logging.Logger = field(
        init=False,
        default_factory=lambda: logging.getLogger(
            f"[VolumeControl {platform.system()}]"
        ),
    )
    platform_name: str = field(init=False, default_factory=platform.system)
    input_device: AudioDevice = field(init=False)
    output_device: AudioDevice = field(init=False)

    @abstractmethod
    def set_output_volume(self, volume: int) -> bool:
        """Set the output volume to the provided value between 0 (minimum volume) and 100 (maximum volume)."""
        pass

    @abstractmethod
    def get_output_volume(self) -> int:
        """Get the output volume as a value between 0 (minimum volume) and 100 (maximum volume)."""
        pass

    @abstractmethod
    def set_input_volume(self, volume: int) -> bool:
        """Set the input volume to the provided value between 0 (minimum volume) and 100 (maximum volume)."""
        pass

    @abstractmethod
    def get_input_volume(self) -> int:
        """Get the input volume as a value between 0 (minimum volume) and 100 (maximum volume)."""
        pass


def create_volume_control() -> VolumeControl:
    """Return the correct VolumeControl subclass for the current platform.

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
        from .volume_control_linux import VolumeControlLinux

        return VolumeControlLinux()
    elif system == "Windows":
        from .volume_control_windows import VolumeControlWindows

        return VolumeControlWindows()
    else:
        raise RuntimeError(f"Unsupported platform for volume control: {system}")
