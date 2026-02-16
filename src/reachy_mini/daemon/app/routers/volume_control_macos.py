"""Volume control implementation for macOS systems."""

import ctypes
from ctypes import POINTER, Structure, byref, c_float, c_int, c_uint32, c_void_p
from dataclasses import dataclass, field
from struct import pack, unpack

from .volume_control import SOUND_CARD_NAMES, DeviceType, VolumeControl


def _get_macos_id(macos_four_char_code: str) -> int:
    """Convert a macOS FourCharCode into an integer ID.

    Args:
        macos_four_char_code: A macOS FourCharCode as a string.

    Returns:
        The corresponding integer ID.

    """
    return int(unpack('!I', macos_four_char_code.encode())[0])


def _get_macos_four_char_code(macos_id: int) -> str:
    """Convert an integer ID into a macOS FourCharCode.

    Args:
        macos_id: An integer ID.

    Returns:
        The corresponding macOS FourCharCode as a string.

    """
    return pack('!I', macos_id).decode('ascii')


# Device and scope constants
kAudioObjectSystemObject = 1
kAudioObjectPropertyScopeGlobal = _get_macos_id('glob')
kAudioObjectPropertyElementMaster = 0
kAudioHardwarePropertyDevices = _get_macos_id('dev#')
kAudioDevicePropertyDeviceNameCFString = _get_macos_id('lnam')
kAudioDevicePropertyDeviceName = _get_macos_id('name')

# Volume control constants
kAudioHardwarePropertyDefaultOutputDevice = _get_macos_id('dOut')
kAudioHardwarePropertyDefaultInputDevice = _get_macos_id('dIn ')
kAudioDevicePropertyScopeOutput = _get_macos_id('outp')
kAudioDevicePropertyScopeInput = _get_macos_id('inpt')
kAudioDevicePropertyVolumeScalar = _get_macos_id('volm')


@dataclass
class VolumeControlMacOS(VolumeControl):
    """Volume control class for macOS.
    
    Relies on the macOS CoreAudio framework.
    """

    input_device_id: int = field(init=False)
    output_device_id: int = field(init=False)

    # Define the AudioObjectPropertyAddress structure
    class AudioObjectPropertyAddress(Structure):
        """Structure representing the address of an audio object property in CoreAudio."""

        _fields_ = [
            ('mSelector', c_uint32),
            ('mScope', c_uint32),
            ('mElement', c_uint32),
        ]

    def __post_init__(self) -> None:
        """Initialize the volume control. Loads the CoreAudio framework and defines the function prototypes."""
        # Load CoreAudio framework
        self.coreaudio = ctypes.CDLL('/System/Library/Frameworks/CoreAudio.framework/Versions/A/CoreAudio')

        # Define function prototypes
        self.coreaudio.AudioObjectGetPropertyDataSize.argtypes = [c_uint32, POINTER(self.AudioObjectPropertyAddress), c_uint32, c_void_p, POINTER(c_uint32)]
        self.coreaudio.AudioObjectGetPropertyDataSize.restype = c_int
        
        self.coreaudio.AudioObjectGetPropertyData.argtypes = [c_uint32, POINTER(self.AudioObjectPropertyAddress), c_uint32, c_void_p, POINTER(c_uint32), c_void_p]
        self.coreaudio.AudioObjectGetPropertyData.restype = c_int

        self.coreaudio.AudioObjectSetPropertyData.argtypes = [c_uint32, POINTER(self.AudioObjectPropertyAddress), c_uint32, c_void_p, c_uint32, c_void_p]
        self.coreaudio.AudioObjectSetPropertyData.restype = c_int

        self.coreaudio.AudioObjectHasProperty.argtypes = [c_uint32, POINTER(self.AudioObjectPropertyAddress)]
        self.coreaudio.AudioObjectHasProperty.restype = c_int

        # Initialize audio devices IDs
        # TODO: use a property instead to account for dynamic audio devices
        self.input_device_id, self.output_device_id = self._get_input_output_device_ids()

    def _get_device_name(self, device_id: int) -> str:
        """Get the name of an audio device given its ID.

        Args:
            device_id: The ID of the audio device.

        Returns:
            The name of the audio device.

        """
        property_address = self.AudioObjectPropertyAddress(
            mSelector=kAudioDevicePropertyDeviceName,
            mScope=kAudioObjectPropertyScopeGlobal,
            mElement=kAudioObjectPropertyElementMaster
        )

        property_data_size = c_uint32(0)
        status = self.coreaudio.AudioObjectGetPropertyDataSize(
            device_id,
            byref(property_address),
            0,
            None,
            byref(property_data_size)
        )

        if status != 0:
            return f"Unknown device (id={device_id})"

        device_name = ctypes.create_string_buffer(property_data_size.value)
        status = self.coreaudio.AudioObjectGetPropertyData(
            device_id,
            byref(property_address),
            0,
            None,
            byref(property_data_size),
            byref(device_name)
        )

        if status != 0:
            return f"Unknown device (id={device_id})"

        return device_name.value.decode('utf-8')

    def _get_input_output_device_ids(self) -> tuple[int, int]:
        """Get the input and output audio devices IDs corresponding to the Reachy Mini Audio sound card. If not found, get the default input and output audio devices IDs.
        
        Returns:
            A tuple containing the input and output audio devices IDs: (input_device_id, output_device_id).

        """
        devices = self._get_all_devices()

        for device_id, device_name in devices.items():
            if device_name in SOUND_CARD_NAMES:
                # Input and output devices will appear with the same ID
                return device_id, device_id

        return self._get_default_device_id(DeviceType.INPUT), self._get_default_device_id(DeviceType.OUTPUT)

    def _get_all_devices(self) -> dict[int, str]:
        """Get all available audio devices IDs and names.
        
        Returns:
            A list of dictionaries containing the ID and name of each audio device: [{id: int, name: str}, ...].

        Raises:
            RuntimeError: If AudioObjectGetPropertyDataSize or AudioObjectGetPropertyData fail when getting all audio devices.

        """
        devices: dict[int, str] = {}

        property_address = self.AudioObjectPropertyAddress(
            mSelector=kAudioHardwarePropertyDevices,
            mScope=kAudioObjectPropertyScopeGlobal,
            mElement=kAudioObjectPropertyElementMaster
        )

        property_data_size = c_uint32(0)

        status = self.coreaudio.AudioObjectGetPropertyDataSize(
            kAudioObjectSystemObject,
            byref(property_address),
            0,
            None,
            byref(property_data_size)
        )

        if status != 0:
            raise RuntimeError(f"Could not query audio device list size (CoreAudio error {status})")

        # Determine the number of devices and create an array to hold the device IDs
        device_count = property_data_size.value // ctypes.sizeof(c_uint32)
        device_ids = (c_uint32 * device_count)()

        status = self.coreaudio.AudioObjectGetPropertyData(
            kAudioObjectSystemObject,
            byref(property_address),
            0,
            None,
            byref(property_data_size),
            byref(device_ids)
        )

        if status != 0:
            raise RuntimeError(f"Could not retrieve audio device list (CoreAudio error {status})")

        for device_id in device_ids:
            device_name = self._get_device_name(device_id)
            devices.setdefault(device_id, device_name)
        return devices

    def _get_default_device_id(self, device_type: DeviceType) -> int:
        """Get the default audio device ID for a given device type.
        
        Args:
            device_type: The type of device: INPUT or OUTPUT.

        Returns:
            The default audio device ID for the given device type.

        Raises:
            RuntimeError: If AudioObjectGetPropertyData fails when getting the default audio devices.

        """
        selector = kAudioHardwarePropertyDefaultOutputDevice if device_type == DeviceType.OUTPUT else kAudioHardwarePropertyDefaultInputDevice

        property_address = self.AudioObjectPropertyAddress(
            mSelector=selector,
            mScope=kAudioObjectPropertyScopeGlobal,
            mElement=kAudioObjectPropertyElementMaster
        )
        device_id = c_uint32(0)
        property_data_size = c_uint32(ctypes.sizeof(c_uint32))

        status = self.coreaudio.AudioObjectGetPropertyData(
            kAudioObjectSystemObject,
            byref(property_address),
            0,
            None,
            byref(property_data_size),
            byref(device_id)
        )

        if status != 0:
            raise RuntimeError(f"Could not retrieve the default {device_type.value} device (CoreAudio error {status})")

        return device_id.value

    def _get_device_volume(self, device_id: int, device_type: DeviceType) -> float:
        """Get the volume of an audio device given its ID and type.
        
        Args:
            device_id: The ID of the audio device.
            device_type: The type of device: INPUT or OUTPUT.

        Returns:
            The volume of the audio device as a value between 0 (minimum volume) and 1 (maximum volume). Returns -1.0 if the volume could not be read.

        """
        # Try master channel (0) first, then individual channels (1, 2)
        channels_to_try = [0, 1, 2]  # 0 = master, 1 = left, 2 = right
        volumes = []

        # Get the appropriate scope based on device type
        scope = kAudioDevicePropertyScopeInput if device_type == DeviceType.INPUT else kAudioDevicePropertyScopeOutput

        for channel in channels_to_try:
            property_address = self.AudioObjectPropertyAddress(
                mSelector=kAudioDevicePropertyVolumeScalar,
                mScope=scope,
                mElement=channel
            )

            # Check if this device has the volume scalar property
            has_property = self.coreaudio.AudioObjectHasProperty(device_id, byref(property_address))
            if not has_property:
                continue  # Skip this channel if the property doesn't exist

            volume = c_float(0.0)
            property_data_size = c_uint32(ctypes.sizeof(c_float))

            status = self.coreaudio.AudioObjectGetPropertyData(
                device_id,
                byref(property_address),
                0,
                None,
                byref(property_data_size),
                byref(volume)
            )

            if status == 0:
                volumes.append(volume.value)
                # If we have a master channel (0), use it and stop
                if channel == 0:
                    return volume.value

        if not volumes:
            self.logger.error(f"No volume channels found on device {device_id} — cannot read volume")    
            return -1.0

        # Return average of available channels (if no master channel)
        return sum(volumes) / len(volumes)

    def _set_device_volume(self, device_id: int, device_type: DeviceType, volume: float) -> bool:
        """Set the volume of an audio device given its ID and type.
        
        Args:
            device_id: The ID of the audio device.
            device_type: The type of device: INPUT or OUTPUT.
            volume: The volume to set between 0 (minimum volume) and 1 (maximum volume).

        Returns:
            True if the volume was set successfully, False otherwise.

        """
        # Clamp volume to valid range
        volume = max(0.0, min(1.0, volume))

        # Try master channel (0) first, then individual channels (1, 2)
        channels_to_try = [0, 1, 2]  # 0 = master, 1 = left, 2 = right
        success_count = 0

        # Get the appropriate scope based on device type
        scope = kAudioDevicePropertyScopeInput if device_type == DeviceType.INPUT else kAudioDevicePropertyScopeOutput

        for channel in channels_to_try:
            property_address = self.AudioObjectPropertyAddress(
                mSelector=kAudioDevicePropertyVolumeScalar,
                mScope=scope,
                mElement=channel
            )

            # Check if this device has the volume scalar property
            has_property = self.coreaudio.AudioObjectHasProperty(device_id, byref(property_address))
            if not has_property:
                continue  # Skip this channel if the property doesn't exist

            volume_value = c_float(volume)
            property_data_size = c_uint32(ctypes.sizeof(c_float))

            status = self.coreaudio.AudioObjectSetPropertyData(
                device_id,
                byref(property_address),
                0,
                None,
                property_data_size,
                byref(volume_value)
            )

            if status == 0:
                success_count += 1
                # If we successfully set master channel (0), we're done
                if channel == 0:
                    return True

        if success_count == 0:
            self.logger.error(f"No volume channels found on device {device_id} — cannot set volume")
            return False

        return True

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
