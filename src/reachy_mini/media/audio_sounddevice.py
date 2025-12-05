"""Audio implementation using sounddevice backend."""

import os
import threading
from collections import deque
from typing import Any, List, Optional

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
        self._buffer: List[npt.NDArray[np.float32]] = []
        self._output_device_id = self._get_device_id(
            ["Reachy Mini Audio", "respeaker"], device_io_type="output"
        )
        self._input_device_id = self._get_device_id(
            ["Reachy Mini Audio", "respeaker"], device_io_type="input"
        )

        # Streaming output state
        self._output_chunk_fifo: deque[npt.NDArray[np.float32]] = deque()
        self._output_queued_samples: int = 0
        self._output_tail: Optional[npt.NDArray[np.float32]] = None
        self._output_max_queue_seconds: float = 5.0
        self._output_underflows: int = 0
        self._output_overflows: int = 0
        self._output_lock = threading.Lock()
        self._output_max_queue_samples = int(self.get_output_audio_samplerate() * self._output_max_queue_seconds)

    @property
    def _is_playing(self) -> bool:
        """Check if output stream is active."""
        return self._output_stream is not None and self._output_stream.active

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
        """Push audio samples to the output stream for playback.
        
        Note: Channel conversion is handled by MediaManager before this is called.
        Data should already be in the correct format for the output device.
        """
        if not self._is_playing:
            self.logger.warning("Output stream is not open. Call start_playing() first.")
            return

        # Ensure C-contiguous array
        data = np.ascontiguousarray(data, dtype=np.float32)

        # Prevent unbounded queue growth
        with self._output_lock:
            if self._output_queued_samples + data.shape[0] > self._output_max_queue_samples:
                while self._output_queued_samples + data.shape[0] > self._output_max_queue_samples and len(self._output_chunk_fifo) > 0:
                    dropped = self._output_chunk_fifo.popleft()
                    self._output_queued_samples -= dropped.shape[0]
                self._output_overflows += 1
                self.logger.warning(
                    f"Audio queue overflow ({self._output_queued_samples} samples), dropped old chunks"
                )

            self._output_chunk_fifo.append(data)
            self._output_queued_samples += data.shape[0]


    def _streaming_callback(
        self,
        outdata: npt.NDArray[np.float32],
        frames: int,
        time: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Fill audio output buffer from queued chunks.
        
        Note: Data from MediaManager is already formatted to match output channels.
        """
        if status and status.output_underflow:
            self._output_underflows += 1
            if self._output_underflows % 10 == 1:
                self.logger.debug(f"Audio underflow count: {self._output_underflows}")

        # Zero the output buffer (silence by default)
        outdata[:] = 0.0
        written = 0

        # Drain carryover tail first
        if self._output_tail is not None and self._output_tail.size:
            take = min(frames, self._output_tail.shape[0])
            outdata[:take, :] = self._output_tail[:take]
            written += take
            self._output_tail = self._output_tail[take:] if take < self._output_tail.shape[0] else None

        # Drain FIFO chunks
        while written < frames:
            with self._output_lock:
                if len(self._output_chunk_fifo) == 0:
                    break

                chunk = self._output_chunk_fifo.popleft()
                self._output_queued_samples -= chunk.shape[0]

            need = frames - written
            if chunk.shape[0] <= need:
                outdata[written:written + chunk.shape[0], :] = chunk
                written += chunk.shape[0]
            else:
                outdata[written:frames, :] = chunk[:need]
                self._output_tail = chunk[need:]
                written = frames
                
        # Apply fade-out if we couldn't fill the buffer
        if written > 0 and written < frames:
            remaining = frames - written
            fade_len = min(64, remaining)
            fade_start_val = outdata[written - 1, :]
            fade_curve = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            outdata[written:written + fade_len, :] = fade_start_val * fade_curve[:, np.newaxis] 
                
    def start_playing(self) -> None:
        """Open the audio output stream."""
        if self._is_playing:
            self.stop_playing()
        self._output_stream = sd.OutputStream(
            samplerate=self.get_output_audio_samplerate(),
            device=self._output_device_id,
            dtype="float32",
            callback=self._streaming_callback,
        )
        if self._output_stream is None:
            raise RuntimeError("Failed to open SoundDevice audio output stream.")
        self._output_stream.start()

    def stop_playing(self) -> None:
        """Close the audio output stream."""
        if self._is_playing:
            self._output_stream.stop()
            self._output_stream.close()
            self._output_stream = None

        with self._output_lock:
            self._output_chunk_fifo.clear()
            self._output_queued_samples = 0
            self._output_tail = None

        self.logger.info("Audio output stream closed.")

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