import logging
import re
import subprocess
from dataclasses import dataclass, field

from .volume_control import SOUND_CARD_NAMES, DeviceType, VolumeControl

logger = logging.getLogger(__name__)

# Constants
AUDIO_COMMAND_TIMEOUT = 2  # Timeout in seconds for audio commands

DEFAULT_INPUT_CONTROLS = ["Master","PCM"]
DEFAULT_OUTPUT_CONTROLS = ["Capture","Mic"]
REACHY_MINI_INPUT_CONTROLS = ["Headset"]
REACHY_MINI_OUTPUT_CONTROLS = ["PCM"]

@dataclass
class VolumeControlLinux(VolumeControl):
    """Volume control class for Linux systems.

    Relies on subprocess calls to `aplay`, `arecord` and `amixer` commands for maximum compatibility.
    """

    input_device_id: int | None = field(init=False)
    output_device_id: int | None = field(init=False)

    def __post_init__(self) -> None:
        """Initialize device IDs based on detected audio devices."""
        # TODO: use a property instead to account for dynamic audio devices
        self.input_device_id, self.output_device_id = self._get_input_output_device_ids()
    
    def _get_device_controls(self, device_id: int) -> list[str]:
        """Get ALSA controls of an audio device given its ID.
        
        Args:
            device_id: The ID of the audio device.
    
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
                        # Extract control name between single quotes
                        start = line.find("'") + 1
                        end = line.find("'", start)
                        if start > 0 and end > start:
                            controls.append(line[start:end])
                            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"Failed to list controls for device {device_id} - amixer failed with error: {e}")
            return []
        
        return controls

    def _get_input_output_device_ids(self) -> tuple[int | None, int | None]:
        """Get the input and output audio device IDs corresponding to the Reachy Mini Audio sound card. If not found, returns (None, None) to fall back to default ALSA controls.

        Returns:
            A tuple containing the input and output device IDs: (input_device_id, output_device_id).

        """
        devices = self._get_all_devices()
        
        for device_id, device_name in devices.items():
            if device_name in SOUND_CARD_NAMES:
                return device_id, device_id
        
        return None, None

    def _get_all_devices(self) -> dict[int, str]:
        """Get all available audio devices IDs and names.

        Returns:
            A dictionary containing the ID and name of each audio device: {id: int, name: str, ...}.

        Raises:
            RuntimeError: If aplay or arecord fail when getting all audio devices.

        """
        devices: dict[int, str] = {}
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

    def _build_amixer_get_command(self, device_id: int | None, device_type: DeviceType) -> list[str]:
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

    def _build_amixer_set_command(self, device_id: int | None, device_type: DeviceType, volume: float) -> list[str]:
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

    def _get_device_volume(self, device_id: int | None, device_type: DeviceType) -> float:
        """Get the volume of an audio device given its ID and type.

        Args:
            device_id: The ID of the audio device.
            device_type: The type of device: INPUT or OUTPUT.

        Returns:
            The volume of the audio device as a value between 0 (minimum volume) and 1 (maximum volume). Returns -1.0 if the volume could not be read.

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
                #TODO: add support for other channels ?
                if "Left:" in line and "[" in line:
                    parts = line.split("[")
                    for part in parts:
                        if "%" in part:
                            volume_str = part.split("%")[0]
                            return float(volume_str) / 100.0

        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, subprocess.CalledProcessError) as e:
            logger.error(f"Failed to get volume on device {device_id} - amixer failed with error: {e}")

        return -1.0

    def _set_device_volume(self, device_id: int | None, device_type: DeviceType, volume: float) -> bool:
        """Set the volume of an audio device given its ID and type.

        Args:
            device_id: The ID of the audio device.
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

    def get_information(self) -> dict[str, int | None | list[str]]:
        """Get information about the controlled audio devices.
        
        Returns:
            A dictionary containing the information about the controlled audio devices.

        """
        return {
            "input_device_id": self.input_device_id,
            "output_device_id": self.output_device_id,
            "available_input_controls": self._get_device_controls(self.input_device_id) if self.input_device_id is not None else [],
            "available_output_controls": self._get_device_controls(self.output_device_id) if self.output_device_id is not None else [],
        }
