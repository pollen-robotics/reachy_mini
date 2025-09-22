"""Audio implementation using sounddevice backend."""

import os

import librosa
import numpy as np
import sounddevice as sd

from reachy_mini.utils.constants import ASSETS_ROOT_PATH

from .audio_base import AudioBackend, AudioBase


class SoundDeviceAudio(AudioBase):
    """Audio device implementation using sounddevice."""

    def __init__(
        self,
        frames_per_buffer=1024,
        log_level="INFO",
        device=None,
    ):
        """Initialize the SoundDevice audio device."""
        super().__init__(backend=AudioBackend.SOUNDDEVICE, log_level=log_level)
        self.frames_per_buffer = frames_per_buffer
        self.device = device
        self.stream = None
        self._output_stream = None
        self._buffer = None
        self._device_id = self.get_output_device_id("respeaker")
        self._samplerate = (
            -1
        )  # will be set on first use to avoid issues if device is not present (CI)

    def start_recording(self):
        """Open the audio input stream, using ReSpeaker card if available."""
        self.stream = sd.InputStream(
            blocksize=self.frames_per_buffer,
            device=self._device_id,
            callback=self._callback,
        )
        self._buffer = []
        self.stream.start()
        self.logger.info("SoundDevice audio stream opened.")

    def _callback(self, indata, frames, time, status):
        if status:
            self.logger.warning(f"SoundDevice status: {status}")
        self._buffer.append(indata.copy())

    def get_audio_sample(self):
        """Read audio data from the buffer. Returns numpy array or None if empty."""
        if self._buffer and len(self._buffer) > 0:
            data = np.concatenate(self._buffer, axis=0)
            self._buffer.clear()
            return data
        self.logger.warning("No audio data available in buffer.")
        return None

    def get_audio_samplerate(self) -> int:
        """Return the samplerate of the audio device."""
        if self._samplerate == -1:
            self._samplerate = int(
                sd.query_devices(self._device_id)["default_samplerate"]
            )
        return self._samplerate

    def stop_recording(self):
        """Close the audio stream and release resources."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.logger.info("SoundDevice audio stream closed.")

    def push_audio_sample(self, data):
        """Push audio data to the output device."""
        if self._output_stream is not None:
            self._output_stream.write(data)
        else:
            self.logger.warning(
                "Output stream is not open. Call start_playing() first."
            )

    def start_playing(self):
        """Open the audio output stream."""
        self._output_stream = sd.OutputStream(
            samplerate=self.get_audio_samplerate(),
            device=self._device_id,
            channels=1,
        )
        self._output_stream.start()

    def stop_playing(self):
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

        data, samplerate_in = librosa.load(
            file_path, sr=self.get_audio_samplerate(), mono=True
        )

        self.logger.debug(f"Playing sound '{file_path}' at {samplerate_in} Hz")

        sd.play(
            data, self.get_audio_samplerate(), device=self._device_id, blocking=False
        )

    def get_output_device_id(self, name_contains: str) -> int:
        """Return the output device id whose name contains the given string (case-insensitive).

        If not found, return the default output device id.
        """
        devices = sd.query_devices()

        for idx, dev in enumerate(devices):
            if name_contains.lower() in dev["name"].lower():
                return idx
        # Return default output device if not found
        self.logger.warning(
            f"No output device found containing '{name_contains}', using default."
        )
        return sd.default.device[1]
