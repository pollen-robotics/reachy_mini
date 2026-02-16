"""Volume control implementation for Windows systems."""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

from .volume_control import SOUND_CARD_NAMES, DeviceType, VolumeControl

logger = logging.getLogger(__name__)

@dataclass
class VolumeControlWindows(VolumeControl):
    """Volume control class for Windows systems.

    Relies on the pycaw library.
    """

    input_device_id: str = field(init=False)
    output_device_id: str = field(init=False)

    def __post_init__(self) -> None:
        """Initialize device IDs based on detected audio devices."""
        # TODO: use a property instead to account for dynamic audio devices
        self.input_device_id, self.output_device_id = self._get_input_output_device_ids()

    def _get_all_devices(self) -> dict[str, str]:
        """Get all available audio devices IDs and names.

        Returns:
            A dictionary mapping device IDs to device names.

        Raises:
            RuntimeError: If the audio device list could not be retrieved.

        """
        devices: dict[str, str] = {}
        try:
            for device in AudioUtilities.GetAllDevices():
                device_id = device.id
                device_name = device.FriendlyName or f"Unknown device (id={device_id})"
                devices[device_id] = device_name
        except Exception as e:
            raise RuntimeError(f"Could not scan audio devices: {e}")
        return devices

    def _get_input_output_device_ids(self) -> tuple[str, str]:
        """Get the input and output audio device IDs corresponding to the Reachy Mini Audio sound card. If not found, get the default input and output audio devices.

        Returns:
            A tuple containing the input and output audio devices: (input_device, output_device).

        """
        devices = self._get_all_devices()

        for device_id, device_name in devices.items():
            if device_name in SOUND_CARD_NAMES:
                return device_id, device_id

        return self._get_default_device_id(DeviceType.INPUT), self._get_default_device_id(DeviceType.OUTPUT)

    def _get_default_device_id(self, device_type: DeviceType) -> str:
        """Get the default audio device for a given device type.

        Args:
            device_type: The type of device: INPUT or OUTPUT.

        Returns:
            The default audio device for the given device type.

        Raises:
            RuntimeError: If the default audio device could not be retrieved.

        """
        try:
            enumerator = AudioUtilities.GetDeviceEnumerator()
            if device_type == DeviceType.INPUT:
                device = enumerator.GetDefaultAudioEndpoint(1, 1)  # eCapture, eMultimedia
            else:
                device = enumerator.GetDefaultAudioEndpoint(0, 1)  # eRender, eMultimedia
            return str(device.GetId())
        except Exception as e:
            raise RuntimeError(f"Failed to get default {device_type.value} device: {e}")

    @staticmethod
    def _get_device_volume_interface(device_id: str) -> Optional[Any]:
        """Get the IAudioEndpointVolume interface for a device.

        Args:
            device_id: The endpoint ID string of the audio device.

        Returns:
            The IAudioEndpointVolume interface, or None if the device is invalid.

        Raises:
            RuntimeError: If the volume interface could not be retrieved.

        """
        try:
            enumerator = AudioUtilities.GetDeviceEnumerator()
            device = enumerator.GetDevice(device_id)
            return device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        except Exception as e:
            raise RuntimeError(f"Failed to get volume interface for device {device_id}: {e}")

    def _get_device_volume(self, device_id: str, device_type: DeviceType) -> float:
        """Get the volume of an audio device.

        Args:
            device_id: The endpoint ID string of the audio device.
            device_type: The type of device: INPUT or OUTPUT.

        Returns:
            The volume as a value between 0 (minimum volume) and 1 (maximum volume). Returns -1.0 if the volume could not be read.

        """
        volume_interface = self._get_device_volume_interface(device_id)
        if volume_interface is None:
            return -1.0

        try:
            volume_db = volume_interface.GetMasterVolumeLevel()
            min_db, max_db, _ = volume_interface.GetVolumeRange()

            if max_db == min_db:
                return 1.0  # Avoid division by zero

            linear_volume = (volume_db - min_db) / (max_db - min_db)
            return float(max(0.0, min(1.0, linear_volume)))
        except Exception as e:
            logger.error(f"Failed to get volume on {device_type.value} device: {e}")
            return -1.0

    def _set_device_volume(self, device_id: str, device_type: DeviceType, volume: float) -> bool:
        """Set the volume of an audio device.

        Args:
            device_id: The endpoint ID string of the audio device.
            device_type: The type of device: INPUT or OUTPUT.
            volume: The volume to set between 0 (minimum volume) and 1 (maximum volume).

        Returns:
            True if the volume was set successfully, False otherwise.

        """
        volume_interface = self._get_device_volume_interface(device_id)
        if volume_interface is None:
            return False

        # Clamp volume to valid range
        volume = max(0.0, min(1.0, volume))

        try:
            min_db, max_db, _ = volume_interface.GetVolumeRange()
            db_volume = min_db + (volume * (max_db - min_db))
            volume_interface.SetMasterVolumeLevel(db_volume, None)
            return True
        except Exception as e:
            logger.error(f"Failed to set volume on {device_type.value} device: {e}")
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
