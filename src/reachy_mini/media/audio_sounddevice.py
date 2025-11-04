"""Audio implementation using sounddevice backend."""

import os
import threading
import time
from collections import deque
from queue import Queue, Empty
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
        super().__init__(backend=AudioBackend.SOUNDDEVICE, log_level=log_level)
        self.frames_per_buffer = frames_per_buffer
        self.stream = None
        self._output_stream = None
        self._buffer: List[npt.NDArray[np.float32]] = []

        # Device ids
        self._output_device_id = self.get_output_device_id("respeaker")
        self._input_device_id = self.get_input_device_id("respeaker")

        self._streaming_active = False
        self._chunk_fifo: deque[npt.NDArray[np.float32]] = deque()
        self._queued_samples: int = 0
        self._tail: Optional[npt.NDArray[np.float32]] = None
        self._target_buffer_ms: int = 120  # tune 80–200 for stability/latency tradeoff
        self._underflows: int = 0
        self._overflows: int = 0

    def start_recording(self) -> None:
        """Open the audio input stream, using ReSpeaker card if available."""
        self.stream = sd.InputStream(
            blocksize=self.frames_per_buffer,
            device=self._input_device_id,
            callback=self._callback,
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="float32",
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

    def get_audio_sample(self) -> Optional[npt.NDArray[np.float32]]:
        """Read audio data from the buffer. Returns numpy array or None if empty."""
        if self._buffer and len(self._buffer) > 0:
            data: npt.NDArray[np.float32] = np.concatenate(self._buffer, axis=0)
            self._buffer.clear()
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
        """Push PCM mono float32 audio into the output FIFO."""
        if not self._streaming_active or self._output_stream is None:
            self.logger.warning("Output stream is not active. Call start_playing() first.")
            return

        # Ensure shape (n,) float32 mono
        if data.ndim == 2:
            if data.shape[1] > 1:
                data = np.mean(data, axis=1)
            else:
                data = data[:, 0]
        data = np.asarray(data, dtype=np.float32, order="C")

        # Push into local FIFO; keep sample count
        self._chunk_fifo.append(data)
        self._queued_samples += data.shape[0]

    # --- helper: compute target watermark in samples ---
    def _target_buffer_samples(self) -> int:
        return int(self.SAMPLE_RATE * (self._target_buffer_ms / 1000.0))
        
        # A modest prebuffer (start at 120 ms) smooths bursty input. Tune 80–200 ms depending on your device/OS. Lower = snappier, higher = safer.
       
        # Try self._target_buffer_ms = 150 and/or frames_per_buffer = 512 (lower latency) or 2048 (more stable).
        
        # FYI: On Windows with ReSpeaker, WASAPI shared mode can be finicky. Try larger frames_per_buffer (e.g., 2048) first; if still choppy, bump _target_buffer_ms to ~200 ms.
        
    # --- improved callback: drain multiple chunks + carry-over ---
    def _streaming_callback(
        self,
        outdata: npt.NDArray[np.float32],
        frames: int,
        time: Any,
        status: sd.CallbackFlags,
    ) -> None:
        # Handle PortAudio status flags
        if status:
            # Track under/over flows for debugging
            if status.input_overflow or status.output_underflow:
                if status.output_underflow:
                    self._underflows += 1
                if status.input_overflow:
                    self._overflows += 1
                self.logger.debug(f"Audio status: {status} (uf={self._underflows}, of={self._overflows})")

        # Ensure we don't leave stale data in other channels
        # outdata shape: (frames, channels)
        out = outdata[:, 0]
        out[:] = 0.0  # default to silence; we’ll fill what we can

        # Keep a small prebuffer to avoid stutter right after starts or small gaps
        target = self._target_buffer_samples()

        # ---- read from tail first
        written = 0
        if self._tail is not None and self._tail.size:
            take = min(frames, self._tail.size)
            out[:take] = self._tail[:take]
            written += take
            if take < self._tail.size:
                self._tail = self._tail[take:]
            else:
                self._tail = None

        # ---- then drain FIFO until we fill this block
        while written < frames:
            try:
                if self._queued_samples < target:
                    # Not enough buffered audio—leave remaining zeros (silence)
                    break

                # Pull next chunk from local FIFO (not the Queue; producer pushes here)
                chunk = self._chunk_fifo.popleft()
                self._queued_samples -= chunk.shape[0]

                need = frames - written
                if chunk.shape[0] <= need:
                    out[written:written + chunk.shape[0]] = chunk
                    written += chunk.shape[0]
                else:
                    # Partially consume; keep remainder as tail
                    out[written:frames] = chunk[:need]
                    self._tail = chunk[need:]  # carry over to next callback
                    written = frames

            except IndexError:
                # FIFO empty
                break

    # --- start_playing: reset tracking, explicit dtype ---
    def start_playing(self) -> None:
        self._streaming_active = True
        self._chunk_fifo.clear()
        self._queued_samples = 0
        self._tail = None
        self._underflows = 0
        self._overflows = 0

        self._output_stream = sd.OutputStream(
            samplerate=self.SAMPLE_RATE,
            device=self._output_device_id,
            channels=1,
            dtype="float32",
            callback=self._streaming_callback,
            blocksize=self.frames_per_buffer,
        )
        if self._output_stream is None:
            raise RuntimeError("Failed to open SoundDevice audio output stream.")
        self._output_stream.start()
        self.logger.info("SoundDevice audio output stream opened (callback mode w/ smoothing).")

    def stop_playing(self) -> None:
        self._streaming_active = False
        if self._output_stream is not None:
            try:
                self._output_stream.stop()
            finally:
                self._output_stream.close()
            self._output_stream = None
        self._chunk_fifo.clear()
        self._queued_samples = 0
        self._tail = None
        self.logger.info("SoundDevice audio output stream closed.")

    def play_sound(self, sound_file: str, autoclean: bool = False) -> None:
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
            samplerate=self.SAMPLE_RATE,
            device=self._output_device_id,
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

    def _find_device_id(
        self, name_contains: str, device_type: str
    ) -> int:
        """Find device ID by name and type with fallback logic.

        Args:
            name_contains: Substring to search for in device name (case-insensitive)
            device_type: Either "input" or "output"

        Returns:
            Device index

        Raises:
            RuntimeError: If no device with appropriate channels found
        """
        devices = sd.query_devices()
        channel_key = f"max_{device_type}_channels"

        # First try: Search for device by specific name (e.g., "respeaker")
        for idx, dev in enumerate(devices):
            if (
                name_contains.lower() in dev["name"].lower()
                and dev.get(channel_key, 0) > 0
            ):
                return idx

        # Log warning if device with specific name not found
        self.logger.warning(
            f"No {device_type} device containing '{name_contains}' found. Using first available {device_type} device."
        )

        # Fallback: Return first device with appropriate channels
        for idx, dev in enumerate(devices):
            if dev.get(channel_key, 0) > 0:
                return idx

        raise RuntimeError(
            f"No {device_type} audio device with {device_type} channels found."
        )

    def get_output_device_id(self, name_contains: str) -> int:
        """Return the output device id whose name contains the given string (case-insensitive).

        If not found, return the first available output device.
        """
        return self._find_device_id(name_contains, "output")

    def get_input_device_id(self, name_contains: str) -> int:
        """Return the input device id whose name contains the given string (case-insensitive).

        If not found, return the first available input device.
        """
        return self._find_device_id(name_contains, "input")
