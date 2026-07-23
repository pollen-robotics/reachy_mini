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
- ``clear_player()``
- ``play_sound()``

"""

import logging
import warnings
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
from reachy_mini.media.audio_utils import (
    has_reachymini_asoundrc,
    resolve_speaker_eq_gains,
)
from reachy_mini.media.device_detection import get_audio_device
from reachy_mini.media.gstreamer_utils import get_sample, handle_default_bus_message

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

# Software AEC (webrtcdsp + webrtcechoprobe) constants shared by the audio
# backends. The elements only accept S16LE at a fixed set of rates
# (8000/16000/32000/48000) and must share a probe name to be paired.
AEC_RATE = 48_000
AEC_CHANNELS = 2
AEC_PROBE_NAME = "reachymini_aec_probe"


def make_speaker_eq(logger: logging.Logger) -> Optional[Gst.Element]:
    """Build an ``equalizer-10bands`` from the configured gains, or ``None``.

    Returns ``None`` (caller keeps the direct link) when all gains are zero,
    the resolved output is not the Reachy Mini Audio device, or the element is
    unavailable, so uncalibrated robots (and fallback outputs) stay
    byte-identical. Callers insert it on the speaker branch only, after the
    wobbler tee, so head motion is driven by the uncorrected signal.
    """
    gains = resolve_speaker_eq_gains()
    if not any(gains):
        return None
    # The correction is tuned for the Reachy head shell + its speaker; skip it
    # when playback falls back to a different output (autoaudiosink).
    if not (has_reachymini_asoundrc() or get_audio_device("Sink") is not None):
        logger.info("Speaker EQ skipped: output is not the Reachy Mini Audio device")
        return None
    eq = Gst.ElementFactory.make("equalizer-10bands")
    if eq is None:
        logger.warning("equalizer-10bands unavailable; skipping speaker EQ")
        return None
    for i, gain in enumerate(gains):
        eq.set_property(f"band{i}", float(gain))

    # Wrap eq + a limiter in a float-internal bin. The EQ boosts must run in
    # F32 and be limited *before* the final int conversion, otherwise a boosted
    # peak clips when the sink quantizes it. audiodynamic is memoryless so it
    # cannot overshoot, guaranteeing no digital clipping for any gains. If it is
    # unavailable, fall back to the bare EQ (pre-limiter behavior).
    in_caps = Gst.ElementFactory.make("capsfilter")
    limiter = Gst.ElementFactory.make("audiodynamic")
    out_conv = Gst.ElementFactory.make("audioconvert")
    if in_caps is None or limiter is None or out_conv is None:
        logger.warning("audiodynamic unavailable; speaker EQ runs without a limiter")
        return eq
    in_caps.set_property("caps", Gst.Caps.from_string("audio/x-raw,format=F32LE"))
    # Hard-knee compressor, ratio 0 = a brickwall limiter that clamps anything
    # above the threshold to it, so the output never exceeds full scale.
    limiter.set_property("threshold", 0.9)
    limiter.set_property("ratio", 0.0)

    eq_bin = Gst.Bin.new("speaker_eq")
    for el in (in_caps, eq, limiter, out_conv):
        eq_bin.add(el)
    in_caps.link(eq)
    eq.link(limiter)
    limiter.link(out_conv)
    eq_bin.add_pad(Gst.GhostPad.new("sink", in_caps.get_static_pad("sink")))
    eq_bin.add_pad(Gst.GhostPad.new("src", out_conv.get_static_pad("src")))
    return eq_bin


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
        Gst.init([])
        self.logger = logging.getLogger(type(self).__module__)
        self.logger.setLevel(log_level)
        self._doa = AudioDoA()
        # Next expected PTS for the playback / send appsrc; -1 means
        # "no previous buffer, anchor to running-time on next push".
        self._appsrc_pts: int = -1

    def _push_appsrc_buffer(
        self, data: npt.NDArray[np.float32]
    ) -> Optional[Gst.FlowReturn]:
        """Stamp and push one F32LE chunk into ``self._appsrc``.

        Gap-aware: the first buffer after a start/flush (``_appsrc_pts < 0``)
        or after a gap larger than ``GAP_RESET_NS`` carries the ``DISCONT``
        flag and a PTS anchored to the current running-time, so an
        ``audiomixer`` downstream can align it on the current timeline.
        Follow-up buffers leave PTS/DTS as ``CLOCK_TIME_NONE`` so the
        mixer places them contiguously by byte offset.

        Returns the ``Gst.FlowReturn`` from ``push_buffer``, or ``None``
        if ``self._appsrc`` is not initialized.
        """
        appsrc = getattr(self, "_appsrc", None)
        if appsrc is None:
            return None
        running_time = appsrc.get_current_running_time()
        duration_ns = (int(data.shape[0]) * Gst.SECOND) // self.SAMPLE_RATE
        new_cue = (
            self._appsrc_pts < 0 or running_time > self._appsrc_pts + self.GAP_RESET_NS
        )
        buf = Gst.Buffer.new_wrapped(data.tobytes())
        if new_cue:
            buf.set_flags(Gst.BufferFlags.DISCONT)
            buf.pts = running_time
            buf.dts = running_time
            self._appsrc_pts = running_time + duration_ns
        else:
            self._appsrc_pts += duration_ns
        return appsrc.push_buffer(buf)

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

    def push_audio_sample(self, data: npt.NDArray[np.float32]) -> None:
        """Push audio data to the output appsrc.

        Args:
            data: Audio samples as a float32 array.

        """
        ret = self._push_appsrc_buffer(data)
        if ret is None:
            self.logger.warning(
                "AppSrc is not initialized. Call start_playing() first."
            )
        elif ret != Gst.FlowReturn.OK:
            self.logger.warning(f"push_buffer dropped: {ret}")

    def clear_output_buffer(self) -> None:
        """Use :meth:`clear_player` instead. Deprecated; does nothing."""
        warnings.warn(
            "clear_output_buffer() is deprecated; use clear_player().",
            DeprecationWarning,
            stacklevel=2,
        )
        self.logger.warning("clear_output_buffer() is deprecated; use clear_player().")

    @abstractmethod
    def clear_player(self) -> None:
        """Drop any queued playback audio immediately (barge-in)."""
        ...

    @abstractmethod
    def play_sound(self, sound_file: str) -> None:
        """Play a sound file."""
        ...
