"""Audio implementation using sounddevice backend.

This module provides a cross-platform audio implementation using the sounddevice
library. It supports microphone input, speaker output, and sound file playback
across different operating systems (Windows, macOS, Linux).

The sounddevice backend features:
- Cross-platform compatibility
- Low-latency audio processing
- Support for multiple audio devices
- Sound file playback (WAV, OGG, FLAC, etc.)
- Automatic sample rate and channel conversion
- Thread-safe audio buffer management

Note:
    This class is typically used internally by the MediaManager when the DEFAULT
    backend is selected. Direct usage is possible but usually not necessary.

Example usage via MediaManager:
    >>> from reachy_mini.media.media_manager import MediaManager, MediaBackend
    >>>
    >>> # Create media manager with sounddevice backend (default)
    >>> media = MediaManager(backend=MediaBackend.DEFAULT, log_level="INFO")
    >>>
    >>> # Start audio recording
    >>> media.start_recording()
    >>>
    >>> # Get audio samples
    >>> samples = media.get_audio_sample()
    >>> if samples is not None:
    ...     print(f"Captured {len(samples)} audio samples")
    >>>
    >>> # Play a sound file
    >>> media.play_sound("/path/to/sound.wav")
    >>>
    >>> # Clean up
    >>> media.stop_recording()
    >>> media.close()

"""

import os
import threading
from collections import deque
from typing import Deque, List, Optional

import numpy as np
import numpy.typing as npt
import scipy
import sounddevice as sd
import soundfile as sf

from reachy_mini.utils.constants import ASSETS_ROOT_PATH

from .audio_base import AudioBase

MAX_INPUT_CHANNELS = 4
MAX_INPUT_QUEUE_SECONDS = 60.0


class SoundDeviceAudio(AudioBase):
    """Audio device implementation using sounddevice.

    This class implements the AudioBase interface using the sounddevice library,
    providing cross-platform audio capture and playback capabilities.

    Attributes:
        Inherits all attributes from AudioBase.
        Additionally manages sounddevice streams and audio buffers.

    """

    def __init__(
        self,
        log_level: str = "INFO",
    ) -> None:
        """Initialize the SoundDevice audio device.

        Args:
            log_level (str): Logging level for audio operations.
                          Default: 'INFO'.

        Note:
            This constructor initializes the sounddevice audio system and sets up
            the necessary audio streams for recording and playback.

        """
        super().__init__(log_level=log_level)
        self._input_stream = None
        self._output_stream = None
        self._input_lock = threading.Lock()
        self._output_lock = threading.Lock()
        self._input_buffer: Deque[npt.NDArray[np.float32]] = deque()
        self._output_buffer: List[npt.NDArray[np.float32]] = []
        self._input_max_queue_seconds: float = MAX_INPUT_QUEUE_SECONDS
        self._input_queued_samples: int = 0

        self._output_device_id = self._get_device_id(
            ["Reachy Mini Audio", "respeaker"], device_io_type="output"
        )
        self._input_device_id = self._get_device_id(
            ["Reachy Mini Audio", "respeaker"], device_io_type="input"
        )

        self._logs = {
            "input_underflows": 0,
            "input_overflows": 0,
        }

    @property
    def _input_max_queue_samples(self) -> int:
        return int(self._input_max_queue_seconds * self.get_input_audio_samplerate())

    @property
    def _is_recording(self) -> bool:
        return self._input_stream is not None and self._input_stream.active

    def start_recording(self) -> None:
        """Open the audio input stream, using ReSpeaker card if available.

        See AudioBase.start_recording() for complete documentation.
        """
        if self._is_recording:
            self.stop_recording()

        self._input_stream = sd.InputStream(
            device=self._input_device_id,
            samplerate=self.get_input_audio_samplerate(),
            callback=self._input_callback,
        )
        if self._input_stream is None:
            raise RuntimeError("Failed to open SoundDevice audio input stream.")

        self._input_buffer.clear()
        self._input_queued_samples = 0
        self._input_stream.start()
        self.logger.info("SoundDevice audio input stream opened.")

    def _input_callback(
        self,
        indata: npt.NDArray[np.float32],
        frames: int,
        time: int,
        status: sd.CallbackFlags,
    ) -> None:
        if status and status.input_underflow:
            self._logs["input_underflows"] += 1
            if self._logs["input_underflows"] % 10 == 1:
                self.logger.debug(
                    f"Audio input underflow count: {self._logs['input_underflows']}"
                )

        with self._input_lock:
            if (
                self._input_queued_samples + indata.shape[0]
                > self._input_max_queue_samples
            ):
                while (
                    self._input_queued_samples + indata.shape[0]
                    > self._input_max_queue_samples
                    and len(self._input_buffer) > 0
                ):
                    dropped = self._input_buffer.popleft()
                    self._input_queued_samples -= dropped.shape[0]
                self._logs["input_overflows"] += 1
                self.logger.warning(
                    "Audio input buffer overflowed, dropped old chunks !"
                )
            self._input_buffer.append(indata[:, :MAX_INPUT_CHANNELS].copy())
            self._input_queued_samples += indata.shape[0]

    def get_audio_sample(self) -> Optional[npt.NDArray[np.float32]]:
        """Read audio data from the buffer. Returns numpy array or None if empty.

        See AudioBase.get_audio_sample() for complete documentation.
        """
        with self._input_lock:
            if self._input_buffer and len(self._input_buffer) > 0:
                data: npt.NDArray[np.float32] = np.concatenate(
                    self._input_buffer, axis=0
                )
                self._input_buffer.clear()
                self._input_queued_samples = 0
                return data
        self.logger.debug("No audio data available in buffer.")
        return None

    def get_input_audio_samplerate(self) -> int:
        """Get the input samplerate of the audio device.

        See AudioBase.get_input_audio_samplerate() for complete documentation.
        """
        return int(
            sd.query_devices(self._input_device_id, "input")["default_samplerate"]
        )

    def get_output_audio_samplerate(self) -> int:
        """Get the output samplerate of the audio device.

        See AudioBase.get_output_audio_samplerate() for complete documentation.
        """
        return int(
            sd.query_devices(self._output_device_id, "output")["default_samplerate"]
        )

    def get_input_channels(self) -> int:
        """Get the number of input channels of the audio device.

        See AudioBase.get_input_channels() for complete documentation.
        """
        return min(
            int(sd.query_devices(self._input_device_id, "input")["max_input_channels"]),
            MAX_INPUT_CHANNELS,
        )

    def get_output_channels(self) -> int:
        """Get the number of output channels of the audio device.

        See AudioBase.get_output_channels() for complete documentation.
        """
        return int(
            sd.query_devices(self._output_device_id, "output")["max_output_channels"]
        )

    def stop_recording(self) -> None:
        """Close the audio stream and release resources.

        See AudioBase.stop_recording() for complete documentation.
        """
        if self._is_recording:
            self._input_stream.stop()  # type: ignore[attr-defined]
            self._input_stream.close()  # type: ignore[attr-defined]
            self._input_stream = None
            self.logger.info("SoundDevice audio stream closed.")

    def push_audio_sample(self, data: npt.NDArray[np.float32]) -> None:
        """Push audio data to the output device.

        See AudioBase.push_audio_sample() for complete documentation.
        """
        if self._output_stream is not None:
            with self._output_lock:
                self._output_buffer.append(data.copy())
        else:
            self.logger.warning(
                "Output stream is not open. Call start_playing() first."
            )

    def clear_output_buffer(self) -> None:
        """Clear the output buffer."""
        with self._output_lock:
            self._output_buffer.clear()

    def set_max_output_buffers(self, max_buffers: int) -> None:
        """Set the maximum number of output buffers to queue in the player.

        Args:
            max_buffers (int): Maximum number of buffers to queue.

        """
        self.logger.warning(
            "set_max_output_buffers is not implemented for SoundDeviceAudio."
        )

    def start_playing(self) -> None:
        """Open the audio output stream.

        See AudioBase.start_playing() for complete documentation.
        """
        self.clear_output_buffer()

        if self._output_stream is not None:
            self.stop_playing()
        self._output_stream = sd.OutputStream(
            samplerate=self.get_output_audio_samplerate(),
            device=self._output_device_id,
            callback=self._output_callback,
        )
        if self._output_stream is None:
            raise RuntimeError("Failed to open SoundDevice audio output stream.")
        self._output_stream.start()
        self.logger.info("SoundDevice audio output stream opened.")

    def _output_callback(
        self,
        outdata: npt.NDArray[np.float32],
        frames: int,
        time: int,
        status: sd.CallbackFlags,
    ) -> None:
        """Handle audio output stream callback."""
        if status:
            self.logger.warning(f"SoundDevice output status: {status}")

        with self._output_lock:
            filled = 0
            while filled < frames and self._output_buffer:
                chunk = self._output_buffer[0]

                needed = frames - filled
                available = len(chunk)
                take = min(needed, available)

                outdata[filled : filled + take] = chunk[:take]
                filled += take

                if take < available:
                    # Partial consumption, keep remainder
                    self._output_buffer[0] = chunk[take:]
                else:
                    # Fully consumed this chunk
                    self._output_buffer.pop(0)

            # Only pad with zeros if buffer is truly empty
            if filled < frames:
                outdata[filled:] = 0

    def ensure_chunk_shape(
        self, chunk: npt.NDArray[np.float32], target_shape: tuple[int, ...]
    ) -> npt.NDArray[np.float32]:
        """Ensure chunk has the shape (frames, num_channels) as required by outdata.

        - If chunk is 1D, tile to required num_channels.
        - If chunk is 2D with mismatched channels, use column 0.
        - If chunk is already correct, return as-is.
        """
        num_channels = target_shape[1] if len(target_shape) > 1 else 1
        if chunk.ndim == 1:
            return np.tile(chunk[:, None], (1, num_channels))
        elif chunk.shape[1] != num_channels:
            # Broadcast first channel only
            return np.tile(chunk[:, [0]], (1, num_channels))
        return chunk

    def stop_playing(self) -> None:
        """Close the audio output stream.

        See AudioBase.stop_playing() for complete documentation.
        """
        if self._output_stream is not None:
            self._output_stream.stop()
            self._output_stream.close()
            self._output_stream = None
            self.clear_output_buffer()
            self.logger.info("SoundDevice audio output stream closed.")

    def play_sound(self, sound_file: str) -> None:
        """Play a sound file.

        See AudioBase.play_sound() for complete documentation.

        Args:
            sound_file (str): Path to the sound file to play. May be given relative to the assets directory or as an absolute path.

        """
        if not os.path.exists(sound_file):
            file_path = f"{ASSETS_ROOT_PATH}/{sound_file}"
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Sound file {sound_file} not found in assets directory or given path."
                )
        else:
            file_path = sound_file

        data, samplerate_in = sf.read(file_path, dtype="float32")
        samplerate_out = self.get_output_audio_samplerate()

        if samplerate_in != samplerate_out:
            data = scipy.signal.resample(
                data, int(len(data) * (samplerate_out / samplerate_in))
            )
        data = self.ensure_chunk_shape(data, (-1, self.get_output_channels()))

        self.logger.debug(f"Playing sound '{file_path}' at {samplerate_in} Hz")

        if self._output_stream is not None:
            self.push_audio_sample(data)
        else:
            self.logger.warning(
                "Output stream wasn't open. We are opening it and leaving it open."
            )
            self.start_playing()
            self.push_audio_sample(data)

    def _get_device_id(
        self, names_contains: List[str], device_io_type: str = "output"
    ) -> int:
        """Return the output device id whose name contains the given strings (case-insensitive).

        Args:
            names_contains (List[str]): List of strings that should be contained in the device name.
            device_io_type (str): 'input' or 'output' to specify device type.

        If not found, return the default output device id.

        """
        devices = sd.query_devices()

        for idx, dev in enumerate(devices):
            for name_contains in names_contains:
                if (
                    name_contains.lower() in dev["name"].lower()
                    and dev[f"max_{device_io_type}_channels"] > 0
                ):
                    return idx
        # Return default output device if not found
        self.logger.warning(
            f"No {device_io_type} device found containing '{names_contains}', using default."
        )
        return self._safe_query_device(device_io_type)

    def _safe_query_device(self, kind: str) -> int:
        try:
            return int(sd.query_devices(None, kind)["index"])
        except sd.PortAudioError:
            return (
                int(sd.default.device[1])
                if kind == "input"
                else int(sd.default.device[0])
            )
        except IndexError:
            return 0
