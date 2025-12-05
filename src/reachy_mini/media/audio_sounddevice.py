"""Audio implementation using sounddevice backend."""

import os
import threading
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import scipy
import sounddevice as sd
import soundfile as sf

from reachy_mini.utils.constants import ASSETS_ROOT_PATH

from .audio_base import AudioBase

MAX_INPUT_CHANNELS = 4

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
        self._output_lock = threading.Lock()
        self._input_buffer: List[npt.NDArray[np.float32]] = []
        self._output_buffer: List[npt.NDArray[np.float32]] = []
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
            callback=self._input_callback,
            samplerate=self.get_input_audio_samplerate(),
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

        self._input_buffer.append(indata[:, :MAX_INPUT_CHANNELS].copy()) # Sounddevice callbacks always use 2D arrays. The slicing handles the reshaping and copying of the data.

    def get_audio_sample(self) -> Optional[npt.NDArray[np.float32]]:
        """Read audio data from the buffer. Returns numpy array or None if empty."""
        if self._input_buffer and len(self._input_buffer) > 0:
            data: npt.NDArray[np.float32] = np.concatenate(self._input_buffer, axis=0)
            self._input_buffer.clear()
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

    def get_input_channels(self) -> int:
        """Get the number of input channels of the audio device."""
        return min(
            int(sd.query_devices(self._input_device_id, "input")["max_input_channels"]),
            MAX_INPUT_CHANNELS
        )

    def get_output_channels(self) -> int:
        """Get the number of output channels of the audio device."""
        return int(
            sd.query_devices(self._output_device_id, "output")["max_output_channels"]
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
            if data.ndim > 1:  # convert to mono
                data = np.mean(data, axis=1)
            with self._output_lock:
                self._output_buffer.append(data.copy())
        else:
            self.logger.warning(
                "Output stream is not open. Call start_playing() first."
            )

    def start_playing(self) -> None:
        """Open the audio output stream."""
        self._output_buffer.clear()  # Clear any old data
        if self._output_stream is not None:
            self.stop_playing()
        self._output_stream = sd.OutputStream(
            samplerate=self.get_output_audio_samplerate(),
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
        """Handle audio output stream callback."""
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
        """Play a sound file.

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
        if data.ndim > 1:  # convert to mono
            data = np.mean(data, axis=1)

        self.logger.debug(f"Playing sound '{file_path}' at {samplerate_in} Hz")

        if self._output_stream is not None:
            self.push_audio_sample(data)
        else:
            self.logger.warning("Output stream wasn't open. We are opening it and leaving it open.")
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
