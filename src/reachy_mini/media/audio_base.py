"""Abstract base class for audio backends.

Provides shared audio constants, sample-rate / channel accessors,
``get_audio_sample()``, ``get_input/output_audio_samplerate()``,
``get_input/output_channels()``, ``set_max_output_buffers()``,
Direction of Arrival (``get_DoA()``), and ``cleanup()`` logic so that
``GStreamerAudio`` and ``GstWebRTCClient`` (which inherits from both
``AudioBase`` and ``CameraBase``) don't duplicate them.

Subclasses must implement:
- ``start_recording()``, ``stop_recording()``
- ``start_playing()``, ``stop_playing()``
- ``push_audio_sample()``
- ``play_sound()``

"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import gi
import numpy as np
import numpy.typing as npt

from reachy_mini.media.audio_control_utils import (
    WRITE_SETTLE_SECONDS,
    AudioConfig,
    init_respeaker_usb,
)
from reachy_mini.media.audio_doa import AudioDoA
from reachy_mini.media.gstreamer_utils import get_sample, handle_default_bus_message

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402


class AudioBase(ABC):
    """Abstract audio backend.

    Attributes:
        SAMPLE_RATE: Default sample rate (16 000 Hz — ReSpeaker hardware).
        CHANNELS: Number of audio channels (2 — stereo).
        GAP_RESET_NS: PTS-continuity threshold for ``_compute_pts``.
            If the gap between the next expected PTS and the appsrc's
            current running-time exceeds this value, we treat it as a
            new utterance and re-anchor to running-time.

    """

    SAMPLE_RATE = 16000
    CHANNELS = 2
    GAP_RESET_NS = 200_000_000  # 200 ms

    def __init__(self, log_level: str = "INFO") -> None:
        """Initialize shared audio attributes (DoA helper)."""
        self.logger = logging.getLogger(type(self).__module__)
        self.logger.setLevel(log_level)
        self._doa = AudioDoA()
        # Next expected PTS for the playback / send appsrc; -1 means
        # "no previous buffer, anchor to running-time on next push".
        self._appsrc_pts: int = -1

    def _compute_pts(
        self,
        num_samples: int,
        running_time_ns: int,
        next_pts_ns: int,
    ) -> tuple[int, int, int]:
        """Return ``(pts_ns, duration_ns, next_pts_ns)`` for an appsrc buffer.

        Anchors PTS to ``running_time_ns`` when ``next_pts_ns`` is
        negative (sentinel for "no previous") or the gap is larger
        than ``GAP_RESET_NS``; otherwise continues the previous
        stream's PTS to keep audio contiguous across consecutive
        push calls.
        """
        duration_ns = (num_samples * 1_000_000_000) // self.SAMPLE_RATE
        if next_pts_ns < 0 or running_time_ns > next_pts_ns + self.GAP_RESET_NS:
            pts_ns = running_time_ns
        else:
            pts_ns = next_pts_ns
        return pts_ns, duration_ns, pts_ns + duration_ns

    def _on_bus_message(
        self, bus: Gst.Bus, msg: Gst.Message, pipeline: Gst.Pipeline
    ) -> bool:
        """Delegate to the shared default-bus-message helper.

        Subclasses can override to add custom behaviour, then return
        ``super()._on_bus_message(bus, msg, pipeline)`` to keep the
        default handling.
        """
        return handle_default_bus_message(self.logger, msg, pipeline)

    def get_audio_sample(self) -> Optional[npt.NDArray[np.float32]]:
        """Pull the next recorded audio chunk.

        Returns:
            A float32 array of shape ``(num_samples, 2)`` (stereo), or
            ``None`` if no data is available yet.

        """
        appsink = getattr(self, "_appsink_audio", None)
        if appsink is None:
            return None
        sample = get_sample(appsink, self.logger)
        if sample is None:
            return None
        return np.frombuffer(sample, dtype=np.float32).reshape(-1, 2)

    def get_input_audio_samplerate(self) -> int:
        """Input sample rate in Hz (16 000)."""
        return self.SAMPLE_RATE

    def get_output_audio_samplerate(self) -> int:
        """Output sample rate in Hz (16 000)."""
        return self.SAMPLE_RATE

    def get_input_channels(self) -> int:
        """Return the number of input channels (2 — stereo)."""
        return self.CHANNELS

    def get_output_channels(self) -> int:
        """Return the number of output channels (2 — stereo)."""
        return self.CHANNELS

    def set_max_output_buffers(self, max_buffers: int) -> None:
        """Limit the number of queued playback buffers.

        Args:
            max_buffers: Maximum number of buffers to queue.

        """
        appsrc = getattr(self, "_appsrc", None)
        if appsrc is not None:
            appsrc.set_property("max-buffers", max_buffers)
            appsrc.set_property("leaky-type", 2)  # drop old buffers
        else:
            self.logger.warning(
                "AppSrc is not initialized. Call start_playing() first."
            )

    def get_DoA(self) -> tuple[float, bool] | None:
        """Get the Direction of Arrival (DoA) from the ReSpeaker.

        Returns:
            A tuple ``(angle_radians, speech_detected)`` or ``None``
            if the device is unavailable.

        """
        return self._doa.get_DoA()

    def apply_audio_config(
        self,
        config: AudioConfig,
        *,
        verify: bool = True,
        write_settle_seconds: float = WRITE_SETTLE_SECONDS,
    ) -> bool:
        """Apply caller-provided audio control parameters to the ReSpeaker.

        This opens a short-lived ReSpeaker USB handle, writes each parameter in
        ``config``, and optionally verifies the written values. The SDK does
        not provide default values for these parameters; callers should pass the
        values tuned for their own app.

        Args:
            config: Sequence of ``(parameter_name, values)`` pairs to write.
            verify: When true, read each parameter back after writing it.
            write_settle_seconds: Delay after each write before readback.

        Returns:
            True when all parameters were written and verified successfully.
            False when the ReSpeaker audio board is unavailable or a parameter
            write/readback fails.

        """
        respeaker = init_respeaker_usb()
        if respeaker is None:
            self.logger.warning("ReSpeaker device not found.")
            return False
        try:
            return respeaker.apply_audio_config(
                config,
                verify=verify,
                write_settle_seconds=write_settle_seconds,
            )
        finally:
            respeaker.close()

    def cleanup(self) -> None:
        """Release shared resources (DoA USB device)."""
        self._doa.close()

    @abstractmethod
    def start_recording(self) -> None:
        """Start capturing audio from the microphone."""
        ...

    @abstractmethod
    def stop_recording(self) -> None:
        """Stop the recording pipeline."""
        ...

    @abstractmethod
    def start_playing(self) -> None:
        """Start the playback pipeline."""
        ...

    @abstractmethod
    def stop_playing(self) -> None:
        """Stop the playback pipeline."""
        ...

    @abstractmethod
    def push_audio_sample(self, data: npt.NDArray[np.float32]) -> None:
        """Push audio data to the output."""
        ...

    @abstractmethod
    def play_sound(self, sound_file: str) -> None:
        """Play a sound file."""
        ...
