"""Volume control base class and factory for platform-specific implementations."""

import logging
import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple

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
    """Base class for volume control.

    ``input_device`` / ``output_device`` are resolved lazily from the current
    audio-device selection and re-resolved when it changes.
    """

    logger: logging.Logger = field(
        init=False,
        default_factory=lambda: logging.getLogger(
            f"[VolumeControl {platform.system()}]"
        ),
    )
    platform_name: str = field(init=False, default_factory=platform.system)
    _cached_devices: tuple[AudioDevice, AudioDevice] | None = field(
        init=False, default=None
    )
    _cached_targets: tuple[str | None, str | None] | None = field(
        init=False, default=None
    )

    @abstractmethod
    def _get_input_output_devices(
        self, input_target: str | None, output_target: str | None
    ) -> tuple[AudioDevice, AudioDevice]:
        """Resolve the (input, output) devices for the selected targets."""

    def _selected_targets(self) -> tuple[str | None, str | None]:
        """Return the selected (input, output) device names (in-process, no HTTP)."""
        from reachy_mini.daemon.app.routers.audio_devices import (
            get_local_selected_input,
            get_local_selected_output,
        )

        return get_local_selected_input(), get_local_selected_output()

    @staticmethod
    def _find_device(
        devices: dict[Any, str], selected: str | None
    ) -> tuple[int | str | None, str] | None:
        """Pick a device from a ``{id: name}`` mapping.

        Prefers the selection (case-insensitive substring, either direction),
        then the Reachy Mini card, else ``None`` for the platform default.
        """
        if selected:
            sel = selected.lower()
            for device_id, name in devices.items():
                lowered = name.lower()
                if sel in lowered or lowered in sel:
                    return device_id, name
        for device_id, name in devices.items():
            if any(card in name.lower() for card in SOUND_CARD_NAMES):
                return device_id, name
        return None

    def _resolve(self) -> tuple[AudioDevice, AudioDevice]:
        """Return the cached devices, re-resolving when the selection changes."""
        targets = self._selected_targets()
        cached = self._cached_devices
        if cached is None or self._cached_targets != targets:
            cached = self._get_input_output_devices(*targets)
            self._cached_devices = cached
            self._cached_targets = targets
        return cached

    @property
    def input_device(self) -> AudioDevice:
        """The input (microphone) device currently being controlled."""
        return self._resolve()[0]

    @property
    def output_device(self) -> AudioDevice:
        """The output (speaker) device currently being controlled."""
        return self._resolve()[1]

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


# Lazily-initialised process-wide singleton shared between the REST
# volume router and the backend's WebRTC command handler. Both paths
# must observe the same VolumeControl instance so that a remote volume
# change is immediately reflected on the next REST query and vice
# versa. See reachy_mini/daemon/backend/abstract.py and
# reachy_mini/daemon/app/routers/volume.py for the two callers.
_volume_control: VolumeControl | None = None


def get_volume_control() -> VolumeControl:
    """Return the shared VolumeControl, creating it on first call."""
    global _volume_control  # noqa: PLW0603
    if _volume_control is None:
        _volume_control = create_volume_control()
    return _volume_control
