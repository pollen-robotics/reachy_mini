"""Volume control implementation for Linux systems."""

import logging
import re
import subprocess
from dataclasses import dataclass, field

from .volume_control import SOUND_CARD_NAMES, DeviceType, VolumeControl

logger = logging.getLogger(__name__)

try:
    import pulsectl
    _PULSECTL_AVAILABLE = True
except ImportError:
    _PULSECTL_AVAILABLE = False

# Constants
AUDIO_COMMAND_TIMEOUT = 2  # Timeout in seconds for audio commands

DEFAULT_INPUT_CONTROLS = ["Master", "PCM"]
DEFAULT_OUTPUT_CONTROLS = ["Capture", "Mic"]
REACHY_MINI_INPUT_CONTROLS = ["Headset"]
REACHY_MINI_OUTPUT_CONTROLS = ["PCM"]


@dataclass
class VolumeControlLinux(VolumeControl):
    """Volume control class for Linux systems.

    Uses pulsectl (PulseAudio/PipeWire) when available, falls back to ALSA (amixer/aplay/arecord) otherwise. Not using pyalsaaudio as fallback as the installation will fail if libasound2 is not present.
    """

    input_device_id: int | str | None = field(init=False)
    output_device_id: int | str | None = field(init=False)

    def __post_init__(self) -> None:
        """Initialize device IDs based on detected audio devices."""
        logger.info(f"Using {'pulsectl (PulseAudio/PipeWire)' if _PULSECTL_AVAILABLE else 'amixer (ALSA)'} backend")
        # TODO: use a property instead to account for dynamic audio devices
        self.input_device_id, self.output_device_id = self._get_input_output_device_ids()

    # ---- Dispatch methods ----

    def _get_all_devices(self) -> dict[int | str, str]:
        """Get all available audio devices IDs and names.

        Returns:
            A dictionary mapping device IDs to device names.

        Raises:
            RuntimeError: If the audio device scan fails.

        """
        if _PULSECTL_AVAILABLE:
            return self._pulse_get_all_devices()
        return self._alsa_get_all_devices()

    def _get_input_output_device_ids(self) -> tuple[int | str | None, int | str | None]:
        """Get the input and output audio device IDs corresponding to the Reachy Mini Audio sound card.

        Returns:
            A tuple containing the input and output device IDs: (input_device_id, output_device_id).

        """
        if _PULSECTL_AVAILABLE:
            return self._pulse_get_input_output_device_ids()
        return self._alsa_get_input_output_device_ids()

    def _get_device_volume(self, device_id: int | str | None, device_type: DeviceType) -> float:
        """Get the volume of an audio device given its ID and type.

        Args:
            device_id: The ID of the audio device.
            device_type: The type of device: INPUT or OUTPUT.

        Returns:
            The volume as a value between 0 (minimum) and 1 (maximum). Returns -1.0 if the volume could not be read.

        """
        if _PULSECTL_AVAILABLE:
            return self._pulse_get_device_volume(device_id, device_type)
        return self._alsa_get_device_volume(device_id, device_type)

    def _set_device_volume(self, device_id: int | str | None, device_type: DeviceType, volume: float) -> bool:
        """Set the volume of an audio device given its ID and type.

        Args:
            device_id: The ID of the audio device.
            device_type: The type of device: INPUT or OUTPUT.
            volume: The volume to set between 0 (minimum volume) and 1 (maximum volume).

        Returns:
            True if the volume was set successfully, False otherwise.

        """
        if _PULSECTL_AVAILABLE:
            return self._pulse_set_device_volume(device_id, device_type, volume)
        return self._alsa_set_device_volume(device_id, device_type, volume)

    # ---- PulseAudio/PipeWire (pulsectl) backend ----

    def _pulse_get_all_devices(self, device_type: DeviceType | None = None) -> dict[int | str, str]:
        """Get all available audio devices IDs and names via pulsectl.

        Args:
            device_type: The type of device: INPUT or OUTPUT. If None, returns all devices.

        Returns:
            A dictionary containing the name of each audio device: {name: str, name: str, ...}. Monitor sources and sinks are not included.

        Raises:
            RuntimeError: If pulsectl fails when getting all audio devices.

        """
        devices: dict[int | str, str] = {}
        try:
            with pulsectl.Pulse("reachy-mini") as pulse:
                if device_type == DeviceType.OUTPUT or device_type is None:
                    for sink in pulse.sink_list():
                        if not sink.monitor_source_name:
                                devices[sink.name] = sink.description or f"Unknown device (id={sink.name})"
                if device_type == DeviceType.INPUT or device_type is None:
                    for source in pulse.source_list():
                        if not source.monitor_of_sink_name:
                                devices[source.name] = source.description or f"Unknown device (id={source.name})"
        except Exception as e:
            raise RuntimeError(f"Could not scan audio devices, pulsectl failed: {e}")
        return devices

    def _pulse_get_input_output_device_ids(self) -> tuple[int | str | None, int | str | None]:
        """Get the input and output audio device IDs via pulsectl.

        If not found, falls back to the default sink/source.

        Returns:
            A tuple containing the input and output device IDs: (input_device_id, output_device_id).

        """
        input_devices = self._pulse_get_all_devices(DeviceType.INPUT)
        output_devices = self._pulse_get_all_devices(DeviceType.OUTPUT)

        # Input and output devices will appear with different IDs
        input_device_id, output_device_id = None, None
        for device_id, device_name in input_devices.items():
            if device_name in SOUND_CARD_NAMES:
                input_device_id = device_id
                break
        for device_id, device_name in output_devices.items():
            if device_name in SOUND_CARD_NAMES:
                output_device_id = device_id
                break

        # Fall back to default devices if no matching device found
        if input_device_id is None:
            input_device_id = self._pulse_get_default_device(DeviceType.INPUT)
        if output_device_id is None:
            output_device_id = self._pulse_get_default_device(DeviceType.OUTPUT)

        return input_device_id, output_device_id

    def _pulse_get_default_device(self, device_type: DeviceType) -> str:
        """Get the default audio device ID for a given type via pulsectl.

        Args:
            device_type: The type of device: INPUT or OUTPUT.

        Returns:
            The default audio device ID.

        Raises:
            RuntimeError: If pulsectl fails when getting the default audio device.

        """
        try:
            with pulsectl.Pulse("reachy-mini") as pulse:
                server_info = pulse.server_info()
                if device_type == DeviceType.INPUT:
                    return str(server_info.default_source_name)
                return str(server_info.default_sink_name)
        except Exception as e:
            raise RuntimeError(f"Failed to get default {device_type.value} device via pulsectl: {e}")

    def _pulse_get_device_volume(self, device_id: int | str | None, device_type: DeviceType) -> float:
        """Get the volume of an audio device via pulsectl.

        Args:
            device_id: The audio device name.
            device_type: The type of device: INPUT or OUTPUT.

        Returns:
            The volume as a value between 0 and 1. Returns -1.0 on failure.

        """
        try:
            with pulsectl.Pulse("reachy-mini") as pulse:
                if device_type == DeviceType.INPUT:
                    device = pulse.get_source_by_name(device_id)
                else:
                    device = pulse.get_sink_by_name(device_id)
                return float(pulse.volume_get_all_chans(device))
        except Exception as e:
            logger.error(f"Failed to get volume on device {device_id} - pulsectl error: {e}")
            return -1.0

    def _pulse_set_device_volume(self, device_id: int | str | None, device_type: DeviceType, volume: float) -> bool:
        """Set the volume of an audio device via pulsectl.

        Args:
            device_id: The audio device name.
            device_type: The type of device: INPUT or OUTPUT.
            volume: The volume to set between 0 (minimum volume) and 1 (maximum volume).

        Returns:
            True if the volume was set successfully, False otherwise.

        """
        # Clamp volume to valid range
        volume = max(0.0, min(1.0, volume))
        try:
            with pulsectl.Pulse("reachy-mini") as pulse:
                if device_type == DeviceType.INPUT:
                    device = pulse.get_source_by_name(device_id)
                else:
                    device = pulse.get_sink_by_name(device_id)
                pulse.volume_set_all_chans(device, volume)  #We set all channels to the same volume
            return True
        except Exception as e:
            logger.error(f"Failed to set volume on device {device_id} - pulsectl error: {e}")
            return False

    # ---- ALSA (amixer/aplay/arecord) backend ----

    def _alsa_get_all_devices(self) -> dict[int | str, str]:
        """Get all available audio devices IDs and names via ALSA.

        Returns:
            A dictionary containing the ID and name of each audio device: {id: int, name: str, ...}.

        Raises:
            RuntimeError: If aplay or arecord fail when getting all audio devices.

        """
        devices: dict[int | str, str] = {}
        try:
            scan_result = subprocess.run(
                ["aplay", "-l", ";", "arecord", "-l"],
                capture_output=True,
                text=True,
                timeout=AUDIO_COMMAND_TIMEOUT,
                check=True,
            )
            pattern = re.compile(r"card\s+(\d+):\s+[^[]+\[([^\]]+)\]")
            for line in scan_result.stdout.splitlines():
                match = pattern.search(line)
                if not match:
                    continue
                device_id = int(match.group(1))
                device_name = match.group(2)
                devices.setdefault(device_id, device_name)
            return devices
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
            raise RuntimeError(f"Could not scan audio devices, aplay or arecord failed: {e}")

    def _alsa_get_input_output_device_ids(self) -> tuple[int | str | None, int | str | None]:
        """Get the input and output audio device IDs via ALSA.

        If not found, returns (None, None) to fall back to default ALSA controls.

        Returns:
            A tuple containing the input and output device IDs: (input_device_id, output_device_id).

        """
        devices = self._alsa_get_all_devices()

        for device_id, device_name in devices.items():
            if device_name in SOUND_CARD_NAMES:
                # Input and output devices will appear with the same ID
                return device_id, device_id

        return None, None

    def _build_amixer_get_command(self, device_id: int | str | None, device_type: DeviceType) -> list[str]:
        """Build the amixer command to get the volume of a specific device and control.

        Args:
            device_id: The ID of the audio device. If None, uses the default audio device.
            device_type: The type of device: INPUT or OUTPUT.

        Returns:
            The amixer command to get the volume of the requested device.

        """
        sub_commands = []
        if device_id is None:
            controls = DEFAULT_OUTPUT_CONTROLS if device_type == DeviceType.OUTPUT else DEFAULT_INPUT_CONTROLS
        else:
            controls = REACHY_MINI_OUTPUT_CONTROLS if device_type == DeviceType.OUTPUT else REACHY_MINI_INPUT_CONTROLS
        for control in controls:
            if device_id is not None:
                cmd = f"amixer -c {device_id} sget {control},0"
            else:
                cmd = f"amixer sget {control},0"
            sub_commands.append(cmd)

        full_command = " || ".join(sub_commands)
        return full_command.split(" ")

    def _build_amixer_set_command(self, device_id: int | str | None, device_type: DeviceType, volume: float) -> list[str]:
        """Build the amixer command to set the volume of a specific device and control.

        Args:
            device_id: The ID of the audio device. If None, uses the default audio device.
            device_type: The type of device: INPUT or OUTPUT.
            volume: The volume to set between 0 (minimum volume) and 1 (maximum volume).

        Returns:
            The amixer command to set the volume of the requested device.

        """
        # Convert from 0.0-1.0 to 0-100 and clamp to valid range
        volume_percent = int(volume * 100)
        volume_percent = max(0, min(100, volume_percent))

        sub_commands = []
        if device_id is None:
            controls = DEFAULT_OUTPUT_CONTROLS if device_type == DeviceType.OUTPUT else DEFAULT_INPUT_CONTROLS
        else:
            controls = REACHY_MINI_OUTPUT_CONTROLS if device_type == DeviceType.OUTPUT else REACHY_MINI_INPUT_CONTROLS
        for control in controls:
            # For each control, set the volume for both left and right channels
            for index in [0, 1]:
                if device_id is not None:
                    cmd = f"amixer -c {device_id} sset {control},{index} {volume_percent}%"
                else:
                    cmd = f"amixer sset {control},{index} {volume_percent}%"
                sub_commands.append(cmd)

        full_command = " || ".join(sub_commands)
        return full_command.split(" ")

    def _alsa_get_device_volume(self, device_id: int | str | None, device_type: DeviceType) -> float:
        """Get the volume of an audio device via amixer.

        Args:
            device_id: The ALSA card number.
            device_type: The type of device: INPUT or OUTPUT.

        Returns:
            The volume as a value between 0 and 1. Returns -1.0 on failure.

        """
        try:
            result = subprocess.run(
                self._build_amixer_get_command(device_id, device_type),
                capture_output=True,
                text=True,
                timeout=AUDIO_COMMAND_TIMEOUT,
                check=True,
            )
            for line in result.stdout.splitlines():
                # TODO: add support for other channels ?
                if "Left:" in line and "[" in line:
                    parts = line.split("[")
                    for part in parts:
                        if "%" in part:
                            volume_str = part.split("%")[0]
                            return float(volume_str) / 100.0

        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, subprocess.CalledProcessError) as e:
            logger.error(f"Failed to get volume on device {device_id} - amixer failed with error: {e}")

        return -1.0

    def _alsa_set_device_volume(self, device_id: int | str | None, device_type: DeviceType, volume: float) -> bool:
        """Set the volume of an audio device via amixer.

        Args:
            device_id: The ALSA card number.
            device_type: The type of device: INPUT or OUTPUT.
            volume: The volume to set between 0 (minimum volume) and 1 (maximum volume).

        Returns:
            True if the volume was set successfully, False otherwise.

        """
        try:
            subprocess.run(
                self._build_amixer_set_command(device_id, device_type, volume),
                capture_output=True,
                text=True,
                timeout=AUDIO_COMMAND_TIMEOUT,
                check=True,
            )
            return True

        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, subprocess.CalledProcessError) as e:
            logger.error(f"Failed to set volume on device {device_id} - amixer failed with error: {e}")
            return False

    # ---- Public API ----

    def get_output_volume(self) -> float:
        """Get the output volume.

        Returns:
            The output volume as a value between 0 (minimum volume) and 1 (maximum volume).

        """
        return self._get_device_volume(self.output_device_id, DeviceType.OUTPUT)

    def set_output_volume(self, volume: float) -> bool:
        """Set the output volume.

        Args:
            volume: The volume to set between 0 (minimum volume) and 1 (maximum volume).

        Returns:
            True if the volume was set successfully, False otherwise.

        """
        return self._set_device_volume(self.output_device_id, DeviceType.OUTPUT, volume)

    def get_input_volume(self) -> float:
        """Get the input volume.

        Returns:
            The input volume as a value between 0 (minimum volume) and 1 (maximum volume).

        """
        return self._get_device_volume(self.input_device_id, DeviceType.INPUT)

    def set_input_volume(self, volume: float) -> bool:
        """Set the input volume.

        Args:
            volume: The volume to set between 0 (minimum volume) and 1 (maximum volume).

        Returns:
            True if the volume was set successfully, False otherwise.

        """
        return self._set_device_volume(self.input_device_id, DeviceType.INPUT, volume)

    # ---- Debug / info ----

    def _get_device_controls(self, device_id: int) -> list[str]:
        """Get ALSA controls of an audio device given its ID.

        Args:
            device_id: The ALSA card number.

        Returns:
            A list of ALSA controls.

        """
        controls = []
        try:
            result = subprocess.run(
                ["amixer", "-c", str(device_id), "scontrols"],
                capture_output=True,
                text=True,
                timeout=AUDIO_COMMAND_TIMEOUT,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "Simple mixer control" in line:
                        start = line.find("'") + 1
                        end = line.find("'", start)
                        if start > 0 and end > start:
                            controls.append(line[start:end])

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"Failed to list controls for device {device_id} - amixer failed with error: {e}")
            return []

        return controls

    def get_information(self) -> dict[str, int | str | None | list[str] | bool]:
        """Get information about the controlled audio devices.

        Returns:
            A dictionary containing the information about the controlled audio devices.

        """
        info: dict[str, int | str | None | list[str] | bool] = {
            "backend": "pulsectl" if _PULSECTL_AVAILABLE else "alsa",
            "input_device_id": self.input_device_id,
            "output_device_id": self.output_device_id,
        }
        if not _PULSECTL_AVAILABLE and isinstance(self.input_device_id, int):
            info["available_input_controls"] = self._get_device_controls(self.input_device_id)
        if not _PULSECTL_AVAILABLE and isinstance(self.output_device_id, int):
            info["available_output_controls"] = self._get_device_controls(self.output_device_id)
        return info