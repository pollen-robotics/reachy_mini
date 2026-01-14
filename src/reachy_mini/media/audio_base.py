"""Base classes for audio implementations.

The audio implementations support various backends and provide a unified
interface for audio input/output. This module defines the abstract base class
that all audio implementations should inherit from, ensuring consistent
API across different audio backends.

Available backends include:
- SoundDevice: Cross-platform audio backend using sounddevice library
- GStreamer: GStreamer-based audio backend for advanced audio processing
- WebRTC: WebRTC-based audio for real-time communication

"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import numpy.typing as npt

from reachy_mini.media.audio_control_utils import ReSpeaker, init_respeaker_usb


class AudioBase(ABC):
    """Abstract class for opening and managing audio devices.

    This class defines the interface that all audio implementations must follow.
    It provides common audio parameters and methods for managing audio devices,
    including microphone input and speaker output functionality.

    Attributes:
        SAMPLE_RATE (int): Default sample rate for audio operations (16000 Hz).
        CHANNELS (int): Default number of audio channels (2 for stereo).
        logger (logging.Logger): Logger instance for audio-related messages.
        _respeaker (Optional[ReSpeaker]): ReSpeaker microphone array device handler.

    """

    SAMPLE_RATE = 16000  # respeaker samplerate
    CHANNELS = 2  # respeaker channels

    def __init__(self, log_level: str = "INFO") -> None:
        """Initialize the audio device.

        Args:
            log_level (str): Logging level for audio operations.
                          Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
                          Default: 'INFO'.

        Note:
            This constructor initializes the logging system and attempts to detect
            and initialize the ReSpeaker microphone array if available.

        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self._respeaker: Optional[ReSpeaker] = init_respeaker_usb()

    def __del__(self) -> None:
        """Destructor to ensure resources are released."""
        if self._respeaker:
            self._respeaker.close()

    @abstractmethod
    def start_recording(self) -> None:
        """Start recording audio.

        This method should initialize the audio recording system and prepare
        it to capture audio data. After calling this method, get_audio_sample()
        should be able to retrieve recorded audio data.

        Note:
            Implementations should handle any necessary resource allocation and
            error checking. If recording cannot be started, implementations should
            log appropriate error messages.

        Raises:
            RuntimeError: If audio recording cannot be started due to hardware
                        or configuration issues.

        """
        pass

    @abstractmethod
    def get_audio_sample(self) -> Optional[npt.NDArray[np.float32]]:
        """Read audio data from the device. Returns the data or None if error.

        Returns:
            Optional[npt.NDArray[np.float32]]: A numpy array containing audio samples
            in float32 format, or None if no data is available or an error occurred.

            The array shape is typically (num_samples,) for mono or
            (num_samples, num_channels) for multi-channel audio.

        Note:
            This method should be called after start_recording() has been called.
            The sample rate and number of channels can be obtained via
            get_input_audio_samplerate() and get_input_channels() respectively.

        Example:
            ```python
            audio.start_recording()
            samples = audio.get_audio_sample()
            if samples is not None:
                print(f"Got {len(samples)} audio samples")
            ```

        """
        pass

    def get_input_audio_samplerate(self) -> int:
        """Get the input samplerate of the audio device.

        Returns:
            int: The sample rate in Hz at which audio is being captured.
                Default is 16000 Hz.

        Note:
            This value represents the number of audio samples captured per second
            for each channel.

        """
        return self.SAMPLE_RATE

    def get_output_audio_samplerate(self) -> int:
        """Get the output samplerate of the audio device.

        Returns:
            int: The sample rate in Hz at which audio is being played back.
                Default is 16000 Hz.

        Note:
            This value represents the number of audio samples played per second
            for each channel.

        """
        return self.SAMPLE_RATE

    def get_input_channels(self) -> int:
        """Get the number of input channels of the audio device.

        Returns:
            int: The number of audio input channels (e.g., 1 for mono, 2 for stereo).
                Default is 2 channels.

        Note:
            For the ReSpeaker microphone array, this typically returns 2 channels
            representing the stereo microphone configuration.

        """
        return self.CHANNELS

    def get_output_channels(self) -> int:
        """Get the number of output channels of the audio device.

        Returns:
            int: The number of audio output channels (e.g., 1 for mono, 2 for stereo).
                Default is 2 channels.

        Note:
            This determines how audio data should be formatted when passed to
            push_audio_sample() method.

        """
        return self.CHANNELS

    @abstractmethod
    def stop_recording(self) -> None:
        """Close the audio device and release resources.

        This method should stop any ongoing audio recording and release
        all associated resources. After calling this method, get_audio_sample()
        should return None until start_recording() is called again.

        Note:
            Implementations should ensure proper cleanup to prevent resource leaks.

        """
        pass

    @abstractmethod
    def start_playing(self) -> None:
        """Start playing audio.

        This method should initialize the audio playback system and prepare
        it to receive audio data via push_audio_sample().

        Note:
            Implementations should handle any necessary resource allocation and
            error checking. If playback cannot be started, implementations should
            log appropriate error messages.

        Raises:
            RuntimeError: If audio playback cannot be started due to hardware
                        or configuration issues.

        """
        pass

    @abstractmethod
    def set_max_output_buffers(self, max_buffers: int) -> None:
        """Set the maximum number of output buffers to queue in the player.

        Args:
            max_buffers (int): Maximum number of buffers to queue.

        """
        pass

    @abstractmethod
    def push_audio_sample(self, data: npt.NDArray[np.float32]) -> None:
        """Push audio data to the output device.

        Args:
            data (npt.NDArray[np.float32]): Audio samples to be played.
                The array should contain float32 values typically in the range [-1.0, 1.0].

                For mono audio: shape should be (num_samples,)
                For stereo audio: shape should be (num_samples, 2)

        Note:
            This method should be called after start_playing() has been called.
            The audio data will be played at the sample rate returned by
            get_output_audio_samplerate().

        """
        pass

    @abstractmethod
    def stop_playing(self) -> None:
        """Stop playing audio and release resources.

        This method should stop any ongoing audio playback and release
        all associated resources. After calling this method, push_audio_sample()
        calls will have no effect until start_playing() is called again.

        Note:
            Implementations should ensure proper cleanup to prevent resource leaks.

        """
        pass

    @abstractmethod
    def play_sound(self, sound_file: str) -> None:
        """Play a sound file.

        Args:
            sound_file (str): Path to the sound file to play.
                Supported formats depend on the specific implementation.

        Note:
            This is a convenience method that handles the complete playback
            of a sound file from start to finish. For more control over
            audio playback, use start_playing(), push_audio_sample(),
            and stop_playing() methods.

        Example:
            ```python
            audio.play_sound("/path/to/sound.wav")
            ```

        """
        pass

    def get_DoA(self) -> tuple[float, bool] | None:
        """Get the Direction of Arrival (DoA) value from the ReSpeaker device.

        The spatial angle is given in radians:
        0 radians is left, π/2 radians is front/back, π radians is right.

        Note: The microphone array requires firmware version 2.1.0 or higher to support this feature.
        The firmware is located in src/reachy_mini/assets/firmware/*.bin.
        Refer to https://wiki.seeedstudio.com/respeaker_xvf3800_introduction/#update-firmware for the upgrade process.

        Returns:
            tuple: A tuple containing the DoA value as a float (radians) and the speech detection as a bool, or None if the device is not found.

        """
        if not self._respeaker:
            self.logger.warning("ReSpeaker device not found.")
            return None

        result = self._respeaker.read("DOA_VALUE_RADIANS")
        if result is None:
            return None
        return float(result[0]), bool(result[1])
