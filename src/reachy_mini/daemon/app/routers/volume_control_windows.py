"""Volume control implementation for Windows systems."""

import logging
import warnings
from dataclasses import dataclass
from typing import Any

from pycaw.pycaw import DEVICE_STATE, AudioUtilities, EDataFlow, ERole

from .volume_control import (
    SOUND_CARD_NAMES,
    AudioDevice,
    AudioDeviceType,
    VolumeControl,
)

logger = logging.getLogger(__name__)


@dataclass
class VolumeControlWindows(VolumeControl):
    """Volume control class for Windows systems.

    Relies on the pycaw library.
    """

    def __post_init__(self) -> None:
        """Initialize device IDs based on detected audio devices."""
        # TODO: use a property instead to account for dynamic audio devices
        self.input_device, self.output_device = self._get_input_output_devices()

    def _get_device_name(self, device_id: str | None) -> str:
        """Get the name of an audio device given its ID.

        Args:
            device_id: The endpoint ID string of the audio device.

        Returns:
            The name of the audio device, or "unknown" if not found.

        """
        if device_id is None:
            return "unknown"
        all_devices = {
            **self._get_all_devices(AudioDeviceType.INPUT),
            **self._get_all_devices(AudioDeviceType.OUTPUT),
        }
        return all_devices.get(device_id, f"Unknown device (id={device_id})")

    def _get_all_devices(
        self, device_type: AudioDeviceType | None = None
    ) -> dict[str, str]:
        """Get all available audio devices IDs and names.

        Args:
            device_type: The type of device: INPUT or OUTPUT. If None, returns all devices. Inactive devices are not included.

        Returns:
            A dictionary mapping device IDs to device names.

        Raises:
            RuntimeError: If the audio device list could not be retrieved.

        """
        if device_type == AudioDeviceType.INPUT:
            data_flow = EDataFlow.eCapture.value
        elif device_type == AudioDeviceType.OUTPUT:
            data_flow = EDataFlow.eRender.value
        else:
            data_flow = EDataFlow.eAll.value

        devices: dict[str, str] = {}
        try:
            with warnings.catch_warnings():
                # suppress COMError warnings
                warnings.simplefilter("ignore", UserWarning)
                for device in AudioUtilities.GetAllDevices(
                    data_flow=data_flow,
                    device_state=DEVICE_STATE.ACTIVE.value,  # only include active devices
                ):
                    device_id = device.id
                    devices[device_id] = (
                        device.FriendlyName or f"Unknown device (id={device_id})"
                    )
        except Exception as e:
            raise RuntimeError(
                f"Could not scan audio devices, pycaw failed with error: {e}"
            )
        return devices

    def _get_input_output_devices(self) -> tuple[AudioDevice, AudioDevice]:
        """Get the input and output audio devices corresponding to the Reachy Mini Audio sound card.

        If not found, falls back to the default input and output audio devices.

        Returns:
            A tuple of two AudioDevice: (input_device, output_device).

        """
        input_devices = self._get_all_devices(AudioDeviceType.INPUT)
        output_devices = self._get_all_devices(AudioDeviceType.OUTPUT)

        input_device: AudioDevice | None = None
        output_device: AudioDevice | None = None
        for device_id, device_name in input_devices.items():
            if any(
                [sound_card in device_name.lower() for sound_card in SOUND_CARD_NAMES]
            ):
                input_device = AudioDevice(
                    device_id, device_name, AudioDeviceType.INPUT
                )
                break
        for device_id, device_name in output_devices.items():
            if any(
                [sound_card in device_name.lower() for sound_card in SOUND_CARD_NAMES]
            ):
                output_device = AudioDevice(
                    device_id, device_name, AudioDeviceType.OUTPUT
                )
                break

        if input_device is None:
            default_id = self._get_default_device_id(AudioDeviceType.INPUT)
            input_device = AudioDevice(
                default_id, self._get_device_name(default_id), AudioDeviceType.INPUT
            )
        if output_device is None:
            default_id = self._get_default_device_id(AudioDeviceType.OUTPUT)
            output_device = AudioDevice(
                default_id, self._get_device_name(default_id), AudioDeviceType.OUTPUT
            )

        return input_device, output_device

    @staticmethod
    def _get_default_device_id(device_type: AudioDeviceType) -> str:
        """Get the default audio device ID for a given device type.

        Args:
            device_type: The type of device: INPUT or OUTPUT.

        Returns:
            The default audio device ID.

        Raises:
            RuntimeError: If the default audio device could not be retrieved.

        """
        data_flow = (
            EDataFlow.eCapture
            if device_type == AudioDeviceType.INPUT
            else EDataFlow.eRender
        )
        try:
            enumerator = AudioUtilities.GetDeviceEnumerator()
            device = enumerator.GetDefaultAudioEndpoint(
                data_flow.value, ERole.eMultimedia.value
            )
            return str(device.GetId())
        except Exception as e:
            raise RuntimeError(f"Failed to get default {device_type.value} device: {e}")

    @staticmethod
    def _get_device_volume_interface(device_id: int | str | None) -> Any:
        """Get the IAudioEndpointVolume interface for a device.

        Uses AudioUtilities.CreateDevice() to wrap the raw IMMDevice and access its EndpointVolume property.

        Args:
            device_id: The endpoint ID string of the audio device.

        Returns:
            The IAudioEndpointVolume interface.

        Raises:
            RuntimeError: If the volume interface could not be retrieved.

        """
        try:
            enumerator = AudioUtilities.GetDeviceEnumerator()
            raw_device = enumerator.GetDevice(device_id)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                audio_device = AudioUtilities.CreateDevice(raw_device)
            if audio_device is None:
                raise RuntimeError(f"Could not create AudioDevice for {device_id}")
            return audio_device.EndpointVolume
        except Exception as e:
            raise RuntimeError(
                f"Failed to get volume interface for device {device_id}: {e}"
            )

    def _get_device_volume(self, device: AudioDevice) -> int:
        """Get the volume of an audio device.

        Args:
            device: The audio device.

        Returns:
            The volume as a value between 0 and 100. Returns -1 on failure.

        """
        try:
            volume_interface = self._get_device_volume_interface(device.id)
            volume_db = volume_interface.GetMasterVolumeLevel()
            min_db, max_db, _ = volume_interface.GetVolumeRange()

            if max_db == min_db:
                return 100  # Avoid division by zero

            linear_volume = (volume_db - min_db) / (max_db - min_db)
            return int(round(max(0.0, min(1.0, linear_volume)) * 100))
        except Exception as e:
            logger.error(
                f"Failed to get volume on {device.device_type.value} device: {e}"
            )
            return -1

    def _set_device_volume(self, device: AudioDevice, volume: int) -> bool:
        """Set the volume of an audio device.

        Args:
            device: The audio device.
            volume: The volume to set between 0 (minimum volume) and 100 (maximum volume).

        Returns:
            True if the volume was set successfully, False otherwise.

        """
        # Clamp and convert to 0.0-1.0 for WASAPI dB calculation
        volume_scalar = max(0.0, min(1.0, volume / 100.0))

        try:
            volume_interface = self._get_device_volume_interface(device.id)
            min_db, max_db, _ = volume_interface.GetVolumeRange()
            db_volume = min_db + (volume_scalar * (max_db - min_db))
            volume_interface.SetMasterVolumeLevel(db_volume, None)
            return True
        except Exception as e:
            logger.error(
                f"Failed to set volume on {device.device_type.value} device: {e}"
            )
            return False

    def get_output_volume(self) -> int:
        """Get the output volume.

        Returns:
            The output volume as a value between 0 (minimum volume) and 100 (maximum volume). Returns -1 on failure.

        """
        return self._get_device_volume(self.output_device)

    def set_output_volume(self, volume: int) -> bool:
        """Set the output volume.

        Args:
            volume: The volume to set between 0 (minimum volume) and 100 (maximum volume).

        Returns:
            True if the volume was set successfully, False otherwise.

        """
        return self._set_device_volume(self.output_device, volume)

    def get_input_volume(self) -> int:
        """Get the input volume.

        Returns:
            The input volume as a value between 0 (minimum volume) and 100 (maximum volume). Returns -1 on failure.

        """
        return self._get_device_volume(self.input_device)

    def set_input_volume(self, volume: int) -> bool:
        """Set the input volume.

        Args:
            volume: The volume to set between 0 (minimum volume) and 100 (maximum volume).

        Returns:
            True if the volume was set successfully, False otherwise.

        """
        return self._set_device_volume(self.input_device, volume)
