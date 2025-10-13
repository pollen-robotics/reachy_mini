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

from .audio_base import AudioBackend, AudioBase


class SoundDeviceAudio(AudioBase):
    """Audio device implementation using sounddevice."""

    def __init__(
        self,
        frames_per_buffer: int = 1024,
        log_level: str = "INFO",
    ) -> None:
        """Initialize the SoundDevice audio device."""
        super().__init__(backend=AudioBackend.SOUNDDEVICE, log_level=log_level)
        self.frames_per_buffer = frames_per_buffer
        self.stream = None
        self._output_stream = None
        self._buffer: List[npt.NDArray[np.float32]] = []
        self._device_id = self.get_output_device_id("respeaker")
        self._samplerate = (
            -1
        )  # will be set on first use to avoid issues if device is not present (CI)

    def start_recording(self) -> None:
        """Open the audio input stream, using ReSpeaker card if available."""
        self.stream = sd.InputStream(
            blocksize=self.frames_per_buffer,
            device=self._device_id,
            callback=self._callback,
        )
        if self.stream is None:
            raise RuntimeError("Failed to open SoundDevice audio stream.")
        self._buffer.clear()
        self.stream.start()
        self.logger.info("SoundDevice audio stream opened.")

    def _callback(
        self,
        indata: npt.NDArray[np.float32],
        frames: int,
        time: int,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            self.logger.warning(f"SoundDevice status: {status}")

        self._buffer.append(indata.copy())

    def get_audio_sample(self) -> Optional[bytes | npt.NDArray[np.float32]]:
        """Read audio data from the buffer. Returns numpy array or None if empty."""
        if self._buffer and len(self._buffer) > 0:
            data: npt.NDArray[np.float32] = np.concatenate(self._buffer, axis=0)
            self._buffer.clear()
            return data
        self.logger.debug("No audio data available in buffer.")
        return None

    def get_audio_samplerate(self) -> int:
        """Return the samplerate of the audio device."""
        if self._samplerate == -1:
            self._samplerate = int(
                sd.query_devices(self._device_id)["default_samplerate"]
            )
        return self._samplerate

    def stop_recording(self) -> None:
        """Close the audio stream and release resources."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.logger.info("SoundDevice audio stream closed.")

    def push_audio_sample(self, data: bytes) -> None:
        """Push audio data to the output device."""
        if self._output_stream is not None:
            self._output_stream.write(data)
        else:
            self.logger.warning(
                "Output stream is not open. Call start_playing() first."
            )

    def start_playing(self) -> None:
        """Open the audio output stream."""
        self._output_stream = sd.OutputStream(
            samplerate=self.get_audio_samplerate(),
            device=self._device_id,
            channels=1,
        )
        if self._output_stream is None:
            raise RuntimeError("Failed to open SoundDevice audio output stream.")
        self._output_stream.start()

    def stop_playing(self) -> None:
        """Close the audio output stream."""
        if self._output_stream is not None:
            self._output_stream.stop()
            self._output_stream.close()
            self._output_stream = None
            self.logger.info("SoundDevice audio output stream closed.")

    def play_sound(self, sound_file: str, autoclean: bool = False) -> None:
        """Play a sound file from the assets directory or a given path using sounddevice and soundfile."""
        file_path = f"{ASSETS_ROOT_PATH}/{sound_file}"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Sound file {file_path} not found.")

        data, samplerate_in = sf.read(file_path, dtype="float32")

        if samplerate_in != self.get_audio_samplerate():
            data = scipy.signal.resample(
                data, int(len(data) * (self.get_audio_samplerate() / samplerate_in))
            )
        if data.ndim > 1:  # convert to mono
            data = np.mean(data, axis=1)

        self.logger.debug(f"Playing sound '{file_path}' at {samplerate_in} Hz")

        self.stop_playing()
        start = [0]  # using list to modify in callback
        length = len(data)

        def callback(
            outdata: npt.NDArray[np.float32],
            frames: int,
            time: Any,  # cdata 'struct PaStreamCallbackTimeInfo *
            status: sd.CallbackFlags,
        ) -> None:
            """Actual playback."""
            if status:
                self.logger.warning(f"SoundDevice output status: {status}")

            end = start[0] + frames
            if end > length:
                # Fill the output buffer with the audio data, or zeros if finished
                outdata[: length - start[0], 0] = data[start[0] :]
                outdata[length - start[0] :, 0] = 0
                raise sd.CallbackStop()
            else:
                outdata[:, 0] = data[start[0] : end]
            start[0] = end

        event = threading.Event()

        self._output_stream = sd.OutputStream(
            samplerate=self.get_audio_samplerate(),
            device=self._device_id,
            channels=1,
            callback=callback,
            finished_callback=event.set,  # release the device when done
        )
        if self._output_stream is None:
            raise RuntimeError("Failed to open SoundDevice audio output stream.")
        self._output_stream.start()

        def _clean_up_thread() -> None:
            """Thread to clean up the output stream after playback.

            The daemon may play sound but should release the audio device.
            """
            event.wait()
            timeout = 5  # seconds
            waited = 0
            while (
                self._output_stream is not None
                and self._output_stream.active
                and waited < timeout
            ):
                time.sleep(0.1)
                waited += 0.1
            self.stop_playing()

        if autoclean:
            threading.Thread(
                target=_clean_up_thread,
                daemon=True,
            ).start()

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
        return int(sd.default.device[1])
