"""Audio implementation using sounddevice backend."""

import os
import threading
import time
from typing import Any, List, Optional

import numpy as np
import numpy.typing as npt
import scipy
import sounddevice as sd
import soundfile as sf

from reachy_mini.utils.constants import ASSETS_ROOT_PATH

from .audio_base import AudioBase


class SoundDeviceAudio(AudioBase):
    """Audio device implementation using sounddevice."""

    def __init__(
        self,
        frames_per_buffer: int = 256,
        log_level: str = "INFO",
    ) -> None:
        """Initialize the SoundDevice audio device."""
        super().__init__(log_level=log_level)
        self.frames_per_buffer = frames_per_buffer
        self.stream = None
        self._output_stream = None
        self._output_lock = threading.Lock()
        self._input_buffer: List[npt.NDArray[np.float32]] = []
        self._output_buffer: List[npt.NDArray[np.float32]] = []
        self._output_device_id = self.get_output_device_id("respeaker")
        self._input_device_id = self.get_input_device_id("respeaker")

    def start_recording(self) -> None:
        """Open the audio input stream, using ReSpeaker card if available."""
        self.stream = sd.InputStream(
            blocksize=self.frames_per_buffer,
            device=self._input_device_id,
            callback=self._input_callback,
            samplerate=self.SAMPLE_RATE,
        )
        if self.stream is None:
            raise RuntimeError("Failed to open SoundDevice audio stream.")
        self._input_buffer.clear()
        self.stream.start()
        self.logger.info("SoundDevice audio stream opened.")

    def _input_callback(
        self,
        indata: npt.NDArray[np.float32],
        frames: int,
        time: int,
        status: sd.CallbackFlags,
    ) -> None:
        # TODO: Handle OOM for never cleaning the input buffer
        if status:
            self.logger.warning(f"SoundDevice status: {status}")

        self._input_buffer.append(indata.copy())

    def get_audio_sample(self) -> Optional[npt.NDArray[np.float32]]:
        """Read audio data from the buffer. Returns numpy array or None if empty."""
        if self._input_buffer and len(self._input_buffer) > 0:
            data: npt.NDArray[np.float32] = np.concatenate(self._input_buffer, axis=0)
            self._input_buffer.clear()
            return data
        self.logger.debug("No audio data available in buffer.")
        return None

    def stop_recording(self) -> None:
        """Close the audio stream and release resources."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.logger.info("SoundDevice audio stream closed.")

    def push_audio_sample(self, data: npt.NDArray[np.float32]) -> None:
        """Push audio data to the output device."""
        if self._output_stream is not None:
            with self._output_lock:
                self._output_buffer.append(data.copy())
        else:
            self.logger.warning(
                "Output stream is not open. Call start_playing() first."
            )

    def start_playing(self) -> None:
        """Open the audio output stream."""
        self._output_buffer.clear()  # Clear any old data
        self._output_stream = sd.OutputStream(
            samplerate=self.SAMPLE_RATE,
            device=self._output_device_id,
            channels=1,
            callback=self._output_callback,
            blocksize=self.frames_per_buffer,
        )
        if self._output_stream is None:
            raise RuntimeError("Failed to open SoundDevice audio output stream.")
        self._output_stream.start()

    def _output_callback(
        self,
        outdata: npt.NDArray[np.float32],
        frames: int,
        time: int,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback function for audio output stream."""
        if status:
            self.logger.warning(f"SoundDevice output status: {status}")
        
        with self._output_lock:
            if self._output_buffer:
                # Get the first chunk from the buffer
                chunk = self._output_buffer[0]
                available = len(chunk)
                
                if available >= frames:
                    # We have enough data for this callback
                    outdata[:, 0] = chunk[:frames]
                    # Remove the used portion
                    if available > frames:
                        self._output_buffer[0] = chunk[frames:]
                    else:
                        self._output_buffer.pop(0)
                else:
                    # Not enough data, fill what we can and pad with zeros
                    outdata[:available, 0] = chunk
                    outdata[available:, 0] = 0
                    self._output_buffer.pop(0)
            else:
                # No data available, output silence
                outdata.fill(0)

    def stop_playing(self) -> None:
        """Close the audio output stream."""
        if self._output_stream is not None:
            self._output_stream.stop()
            self._output_stream.close()
            self._output_stream = None
            self.logger.info("SoundDevice audio output stream closed.")

    def play_sound(self, sound_file: str) -> None:
        """Play a sound file from the assets directory or a given path using sounddevice and soundfile."""
        file_path = f"{ASSETS_ROOT_PATH}/{sound_file}"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Sound file {file_path} not found.")

        data, samplerate_in = sf.read(file_path, dtype="float32")

        if samplerate_in != self.SAMPLE_RATE:
            data = scipy.signal.resample(
                data, int(len(data) * (self.SAMPLE_RATE / samplerate_in))
            )
        if data.ndim > 1:  # convert to mono
            data = np.mean(data, axis=1)

        self.logger.debug(f"Playing sound '{file_path}' at {samplerate_in} Hz")

        if self._output_stream is not None:
            self.push_audio_sample(data)
        else:
            self.logger.warning("Output stream wasn't open. We are opening it and leaving it open.")
            self.start_playing()
            self.push_audio_sample(data)

    def get_output_device_id(self, name_contains: str) -> int:
        """Return the output device id whose name contains the given string (case-insensitive).

        If not found, return the default output device id.
        """
        devices = sd.query_devices()

        for idx, dev in enumerate(devices):
            if (
                name_contains.lower() in dev["name"].lower()
                and dev["max_output_channels"] > 0
            ):
                return idx
        # Return default output device if not found
        self.logger.warning(
            f"No output device found containing '{name_contains}', using default."
        )
        return self._safe_query_device("output")

    def get_input_device_id(self, name_contains: str) -> int:
        """Return the input device id whose name contains the given string (case-insensitive).

        If not found, return the default input device id.
        """
        devices = sd.query_devices()

        for idx, dev in enumerate(devices):
            if (
                name_contains.lower() in dev["name"].lower()
                and dev["max_input_channels"] > 0
            ):
                return idx
        # Return default input device if not found
        self.logger.warning(
            f"No input device found containing '{name_contains}', using default."
        )
        return self._safe_query_device("input")

    def _safe_query_device(self, kind: str) -> int:
        try:
            return int(sd.query_devices(None, kind)["index"])
        except sd.PortAudioError:
            return int(sd.default.device[1])
