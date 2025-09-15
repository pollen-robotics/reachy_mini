"""Audio implementation using sounddevice backend."""

import os

import numpy as np
import samplerate as sr
import sounddevice as sd
import soundfile as sf

from reachy_mini.utils.constants import ASSETS_ROOT_PATH

from .audio_base import AudioBackend, AudioBase
from .audio_utils import get_respeaker_card_number


class SoundDeviceAudio(AudioBase):
    """Audio device implementation using sounddevice."""

    def __init__(
        self,
        samplerate=16000,
        channels=1,
        dtype="float32",
        frames_per_buffer=1024,
        log_level="INFO",
        device=None,
    ):
        """Initialize the SoundDevice audio device."""
        super().__init__(backend=AudioBackend.SOUNDDEVICE, log_level=log_level)
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.frames_per_buffer = frames_per_buffer
        self.device = device
        self.stream = None
        self._buffer = None
        self._device_id = self.get_output_device_id("respeaker")
        self._samplerate = int(sd.query_devices(self._device_id)["default_samplerate"])

    def open(self):
        """Open the audio input stream, using ReSpeaker card if available."""
        if self.device is None:
            self.device = get_respeaker_card_number()
            self.logger.info(f"Using audio device: {self.device}")
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.frames_per_buffer,
            device=self.device,
            callback=self._callback,
        )
        self._buffer = []
        self.stream.start()
        self.logger.info("SoundDevice audio stream opened.")

    def _callback(self, indata, frames, time, status):
        if status:
            self.logger.warning(f"SoundDevice status: {status}")
        self._buffer.append(indata.copy())

    def read(self):
        """Read audio data from the buffer. Returns numpy array or None if empty."""
        if self._buffer and len(self._buffer) > 0:
            data = np.concatenate(self._buffer, axis=0)
            self._buffer.clear()
            return data
        return None

    def close(self):
        """Close the audio stream and release resources."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.logger.info("SoundDevice audio stream closed.")

    def play_sound(self, sound_file: str) -> None:
        """Play a sound file from the assets directory or a given path using sounddevice and soundfile."""
        file_path = f"{ASSETS_ROOT_PATH}/{sound_file}"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Sound file {file_path} not found.")

        data, samplerate_in = sf.read(file_path, dtype="float32")
        self.logger.debug(f"Playing sound '{file_path}' at {samplerate_in} Hz")

        if samplerate_in != self._samplerate:
            ratio = self._samplerate / samplerate_in
            data = sr.resample(data, ratio, "sinc_best")

            self.logger.debug(
                f"Resampled audio from {samplerate_in} Hz to {self._samplerate} Hz"
            )

        sd.play(data, self._samplerate, device=self._device_id, blocking=True)

    def get_output_device_id(self, name_contains: str) -> int:
        """Return the output device id whose name contains the given string (case-insensitive).

        If not found, return the default output device id.
        """
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if (
                dev["max_output_channels"] > 0
                and name_contains.lower() in dev["name"].lower()
            ):
                return idx
        # Return default output device if not found
        self.logger.warning(
            f"No output device found containing '{name_contains}', using default."
        )
        return sd.default.device[1]
