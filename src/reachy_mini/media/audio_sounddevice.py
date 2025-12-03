"""Audio implementation using sounddevice backend."""

import os
import threading
from typing import Any, List, Optional

import numpy as np
import numpy.typing as npt
import scipy
import sounddevice as sd
import soundfile as sf

from reachy_mini.utils.constants import ASSETS_ROOT_PATH

from .audio_base import AudioBase

MAX_INPUT_CHANNELS = 4
MAX_OUTPUT_CHANNELS = 4

class SoundDeviceAudio(AudioBase):
    """Audio device implementation using sounddevice."""

    def __init__(
        self,
        log_level: str = "INFO",
    ) -> None:
        """Initialize the SoundDevice audio device."""
        super().__init__(log_level=log_level)
        self.stream = None
        self._output_stream = None
        self._buffer: List[npt.NDArray[np.float32]] = []
        self._output_device_id = self._get_device_id(
            ["Reachy Mini Audio", "respeaker"], device_io_type="output"
        )
        self._input_device_id = self._get_device_id(
            ["Reachy Mini Audio", "respeaker"], device_io_type="input"
        )

    def start_recording(self) -> None:
        """Open the audio input stream, using ReSpeaker card if available."""
        self.stream = sd.InputStream(
            device=self._input_device_id,
            samplerate=self.get_input_audio_samplerate(),
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

        self._buffer.append(indata[:, :MAX_INPUT_CHANNELS]) # Sounddevice callbacks always use 2D arrays. The slicing handles the reshaping and copying of the data.

    def get_audio_sample(self) -> Optional[npt.NDArray[np.float32]]:
        """Read audio data from the buffer. Returns numpy array or None if empty."""
        if self._buffer and len(self._buffer) > 0:
            data: npt.NDArray[np.float32] = np.concatenate(self._buffer, axis=0)
            self._buffer.clear()
            return data
        self.logger.debug("No audio data available in buffer.")
        return None

    def get_input_audio_samplerate(self) -> int:
        """Get the input samplerate of the audio device."""
        return int(
            sd.query_devices(self._input_device_id, "input")["default_samplerate"]
        )

    def get_output_audio_samplerate(self) -> int:
        """Get the output samplerate of the audio device."""
        return int(
            sd.query_devices(self._output_device_id, "output")["default_samplerate"]
        )

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
            if data.ndim > 2 or data.ndim == 0:
                self.logger.warning(f"Audio samples arrays must have at most 2 dimensions and at least 1 dimension, got {data.ndim}")
                return
            
            # Transpose data to match sounddevice channels last convention
            if data.ndim == 2 and data.shape[1] > data.shape[0]:
                data = data.T

            # Fit data to match output stream channels
            output_channels = min(MAX_OUTPUT_CHANNELS, self._output_stream.channels)

            # Mono input to multiple channels output : duplicate to fit
            if data.ndim == 1 and output_channels > 1:
                data = np.column_stack((data,) * output_channels)
            # Lower channels input to higher channels output : reduce to mono and duplicate to fit
            elif data.ndim == 2 and data.shape[1] < output_channels:
                data = np.column_stack((data[:,0],) * output_channels)
            # Higher channels input to lower channels output : crop to fit
            elif data.ndim == 2 and data.shape[1] > output_channels:
                data = data[:, :output_channels]
                
            self._output_stream.write(data)
        else:
            self.logger.warning(
                "Output stream is not open. Call start_playing() first."
            )

    def start_playing(self) -> None:
        """Open the audio output stream."""
        if self._output_stream is not None:
            self.stop_playing()
        self._output_stream = sd.OutputStream(
            samplerate=self.get_output_audio_samplerate(),
            device=self._output_device_id,
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
        """Play a sound file.

        Args:
            sound_file (str): Path to the sound file to play. May be given relative to the assets directory or as an absolute path.
            autoclean (bool): If True, the audio device will be released after the sound is played.

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
        if data.ndim > 1:  # convert to mono
            data = np.mean(data, axis=1)

        self.logger.debug(f"Playing sound '{file_path}' at {samplerate_in} Hz")

        self.stop_playing()
        start = 0  # current position in audio data
        length = len(data)

        def callback(
            outdata: npt.NDArray[np.float32],
            frames: int,
            time: Any,  # cdata 'struct PaStreamCallbackTimeInfo *
            status: sd.CallbackFlags,
        ) -> None:
            """Actual playback."""
            nonlocal start

            if status:
                self.logger.warning(f"SoundDevice output status: {status}")

            end = start + frames
            if end > length:
                # Fill the output buffer with the audio data, or zeros if finished
                outdata[: length - start, 0] = data[start:]
                outdata[length - start :, 0] = 0
                raise sd.CallbackStop()
            else:
                outdata[:, 0] = data[start:end]
            start = end

        stop_event = threading.Event()

        self._output_stream = sd.OutputStream(
            samplerate=samplerate_out,
            device=self._output_device_id,
            channels=1,
            callback=callback,
            finished_callback=stop_event.set,  # release the device when done
        )
        if self._output_stream is None:
            raise RuntimeError("Failed to open SoundDevice audio output stream.")
        self._output_stream.start()

        if autoclean:

            def _clean_up_thread() -> None:
                stop_event.wait()
                self.stop_playing()

            threading.Thread(
                target=_clean_up_thread,
                daemon=True,
            ).start()

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
