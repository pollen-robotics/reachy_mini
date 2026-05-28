"""GStreamer audio backend.

Handles microphone input, speaker output, and sound-file playback using
GStreamer pipelines.  Also provides Direction of Arrival (DoA) estimation
via the ReSpeaker microphone array (see ``AudioDoA``).

Recording pipeline::

    platform_source → queue → audioconvert → audioresample → appsink(F32LE)

Playback pipeline::

    appsrc(F32LE) → audioconvert → audioresample → platform_sink

Platform audio sources / sinks are discovered at runtime:

* **Linux (PipeWire / PulseAudio)**: ``pulsesrc`` / ``pulsesink``
* **Linux (ALSA, Reachy Mini Wireless)**: ``alsasrc`` / ``alsasink``
  with the preconfigured ``reachymini_audio_src`` / ``reachymini_audio_sink``
  devices from ``~/.asoundrc``.
* **Windows**: ``wasapi2src`` / ``wasapi2sink``
* **macOS**: ``osxaudiosrc`` / ``osxaudiosink``
* **Fallback**: ``autoaudiosrc`` / ``autoaudiosink``

The "Reachy Mini Audio" card is located by name via ``Gst.DeviceMonitor``.
If no matching card is found the platform default is used instead.

Software acoustic echo cancellation (no-board fallback)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When no XVF3800 audio board is in the audio path (no Wireless
``~/.asoundrc`` AND no ReSpeaker USB dongle detected), the backend
falls back to the platform default mic + speaker. Without the
hardware AEC of the XMOS chip the OpenAI Realtime / conversation
loop hears its own voice through the laptop speakers and collapses
on feedback.

To keep the simulation / dev workflow usable, this backend inserts
a ``webrtcdsp`` + ``webrtcechoprobe`` pair (from
``gst-plugins-bad``) on top of the record / playback pipelines in
that case. Detection is automatic:

* If ``has_reachymini_asoundrc()`` is true → no extra processing
  (Wireless XMOS handles AEC).
* If ``get_audio_device("Source")`` finds a ReSpeaker card → no
  extra processing (USB XVF3800 handles AEC).
* Otherwise → enable the software AEC stage.

Two environment variables can be set to override the heuristic, for
example to benchmark or troubleshoot:

* ``REACHY_MINI_DISABLE_SW_AEC=1`` force-disables the fallback.
* ``REACHY_MINI_FORCE_SW_AEC=1`` force-enables it (e.g. to compare
  output against the hardware XVF3800 AEC on a board-equipped robot).

Note:
    This class is typically used internally by ``MediaManager`` when the
    ``LOCAL`` backend is selected.  Direct usage is possible but usually
    not necessary.

Example usage via MediaManager::

    from reachy_mini.media.media_manager import MediaManager, MediaBackend

    media = MediaManager(backend=MediaBackend.LOCAL)
    media.start_recording()

    samples = media.get_audio_sample()
    if samples is not None:
        print(f"Captured {len(samples)} audio samples")

    doa = media.get_DoA()
    if doa is not None:
        angle, speech_detected = doa
        print(f"Sound direction: {angle} rad, speech detected: {speech_detected}")

    media.stop_recording()
    media.close()

"""

import os
import platform
import time
from collections.abc import Callable
from threading import Thread
from typing import Optional

import numpy as np
import numpy.typing as npt

from reachy_mini.media.audio_base import AudioBase
from reachy_mini.media.audio_utils import has_reachymini_asoundrc
from reachy_mini.media.device_detection import get_audio_device
from reachy_mini.motion.head_wobbler import HeadWobbler, SpeechOffsets
from reachy_mini.utils.constants import ASSETS_ROOT_PATH

try:
    import gi
except ImportError as e:
    raise ImportError(
        "The 'gi' module is required for GStreamerAudio but could not be imported. "
        "Please check the gstreamer installation."
    ) from e

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")

from gi.repository import GLib, Gst  # noqa: E402


class GStreamerAudio(AudioBase):
    """Audio implementation using GStreamer.

    Extends ``AudioBase`` with two GStreamer-specific helpers:

    - ``clear_output_buffer()``: flush queued playback data without stopping
      the pipeline (no-op by default; useful before refilling the buffer).
    - ``clear_player()``: flush the playback appsrc immediately via GStreamer
      flush events, dropping any queued audio.

    """

    PLAYBACK_SINK_BUFFER_TIME_US = 50_000
    PLAYBACK_SINK_LATENCY_TIME_US = 5_000

    # Shared name that ties the recording-side `webrtcdsp` to the
    # playback-side `webrtcechoprobe`. Both elements communicate via
    # a process-wide registry indexed by this string, so the value
    # just needs to be stable and unique within the process.
    _AEC_PROBE_NAME = "reachymini_aec_probe"

    def __init__(self, log_level: str = "INFO") -> None:
        """Initialize recording and playback pipelines.

        Args:
            log_level: Logging level for audio operations.
                Options: ``'DEBUG'``, ``'INFO'``, ``'WARNING'``, ``'ERROR'``,
                ``'CRITICAL'``.

        """
        super().__init__(log_level=log_level)

        self._head_wobbler: Optional[HeadWobbler] = None

        Gst.init([])
        self._loop = GLib.MainLoop()
        self._thread_bus_calls = Thread(target=lambda: self._loop.run(), daemon=True)
        self._thread_bus_calls.start()

        self._sw_aec_enabled = self._resolve_sw_aec_enabled()

        self._pipeline_record = Gst.Pipeline.new("audio_recorder")
        self._init_pipeline_record(self._pipeline_record)
        self._bus_record = self._pipeline_record.get_bus()
        self._bus_record.add_watch(
            GLib.PRIORITY_DEFAULT, self._on_bus_message, self._pipeline_record
        )

        self._playbin: Optional[Gst.Element] = None
        self._pipeline_playback = Gst.Pipeline.new("audio_player")
        self._init_pipeline_playback(self._pipeline_playback)
        self._bus_playback = self._pipeline_playback.get_bus()
        self._bus_playback.add_watch(
            GLib.PRIORITY_DEFAULT, self._on_bus_message, self._pipeline_playback
        )

    # ------------------------------------------------------------------
    # Software AEC fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _sw_aec_env_override() -> Optional[bool]:
        """Return the env-var override for the software AEC, or None.

        ``REACHY_MINI_DISABLE_SW_AEC`` takes precedence over
        ``REACHY_MINI_FORCE_SW_AEC`` so an explicit disable always wins;
        anything else falls back to the auto-detection heuristic.
        """
        truthy = {"1", "true", "yes", "on"}
        if os.environ.get("REACHY_MINI_DISABLE_SW_AEC", "").lower() in truthy:
            return False
        if os.environ.get("REACHY_MINI_FORCE_SW_AEC", "").lower() in truthy:
            return True
        return None

    def _resolve_sw_aec_enabled(self) -> bool:
        """Decide whether to insert a `webrtcdsp` + `webrtcechoprobe` pair.

        See the module docstring for the full rationale. Short version:
        return ``True`` only when neither hardware AEC (Wireless XMOS
        loopback via ``.asoundrc`` or USB XVF3800 dongle) is present,
        unless an environment variable overrides the heuristic.

        Also verifies that both AEC elements are actually present in
        the running GStreamer build — packaging without
        ``gst-plugins-bad`` would otherwise crash pipeline construction
        with a cryptic ``Failed to create GStreamer elements``.
        """
        # Read via the class so tests can fake `self` with a plain
        # `SimpleNamespace`; the helper is a `@staticmethod` and never
        # depends on instance state.
        override = GStreamerAudio._sw_aec_env_override()
        if override is False:
            self.logger.info("Software AEC disabled via REACHY_MINI_DISABLE_SW_AEC.")
            return False

        if override is True:
            self.logger.warning(
                "Software AEC force-enabled via REACHY_MINI_FORCE_SW_AEC. "
                "If a hardware AEC (XMOS / XVF3800) is also active you "
                "will likely double-process and end up with audible "
                "artefacts — this knob is meant for A/B testing."
            )
        else:
            # Auto-detection path. Bail out as soon as we find any
            # hardware AEC in the audio path.
            if has_reachymini_asoundrc():
                self.logger.info(
                    "Wireless `.asoundrc` detected — relying on XMOS "
                    "hardware AEC, skipping software AEC stage."
                )
                return False
            if get_audio_device("Source") is not None:
                self.logger.info(
                    "ReSpeaker / Reachy Mini Audio card detected — "
                    "relying on XVF3800 hardware AEC, skipping software "
                    "AEC stage."
                )
                return False

        for name in ("webrtcdsp", "webrtcechoprobe"):
            if Gst.ElementFactory.find(name) is None:
                self.logger.warning(
                    "Software AEC requested but GStreamer element '%s' "
                    "is unavailable in this build (install "
                    "gst-plugins-bad). Conversation audio will loop "
                    "back through the laptop mic — use a headset to "
                    "work around it.",
                    name,
                )
                return False

        self.logger.warning(
            "No XVF3800 audio board detected — enabling software AEC "
            "(webrtcdsp + webrtcechoprobe). Quality is best-effort; for "
            "production voice loops use the ReSpeaker USB dongle or a "
            "Wireless robot with the XMOS AEC loopback."
        )
        return True

    def _make_aec_capsfilter(self) -> Gst.Element:
        """Force ``webrtcdsp``-friendly caps (S16LE interleaved, 16 kHz).

        `webrtcdsp` accepts either ``S16LE/interleaved`` or
        ``F32LE/non-interleaved``; the rest of our pipeline uses
        ``F32LE/interleaved``, so we pin S16LE on the AEC link and let
        the surrounding ``audioconvert`` elements bridge the format
        gap on both sides.
        """
        capsfilter = Gst.ElementFactory.make("capsfilter")
        capsfilter.set_property(
            "caps",
            Gst.Caps.from_string(
                "audio/x-raw,"
                "format=S16LE,"
                "layout=interleaved,"
                f"rate={self.SAMPLE_RATE},"
                f"channels={self.CHANNELS}"
            ),
        )
        return capsfilter

    def _append_aec_dsp(
        self, pipeline: Gst.Pipeline, src_element: Gst.Element
    ) -> Gst.Element:
        """Append the recording-side AEC DSP chain to ``src_element``.

        Inserts ``audioconvert → capsfilter(S16LE) → audiobuffersplit
        (10 ms) → webrtcdsp(probe=..., delay-agnostic, extended-filter)
        → audioconvert`` and returns the new tail element to which the
        next downstream consumer should link.

        ``audiobuffersplit`` is required because libwebrtc-audio-
        processing operates on fixed 10 ms frames; without it the dsp
        either rejects or stutters depending on the upstream buffer
        cadence. ``delay-agnostic`` + ``extended-filter`` is the
        recommended combo when the mic and reference clocks come from
        different audio devices (which is exactly our case here:
        record pipeline drives off the mic clock, playback pipeline
        off the speaker clock).
        """
        ac_in = Gst.ElementFactory.make("audioconvert")
        capsf = self._make_aec_capsfilter()
        bsplit = Gst.ElementFactory.make("audiobuffersplit")
        # `output-buffer-duration` is a GstFraction. PyGObject's
        # set_property() segfaults when given a `Gst.Fraction(1, 100)`
        # value, so we route through `Gst.util_set_object_arg` which
        # accepts the same property as a string and constructs the
        # underlying GValue C-side.
        Gst.util_set_object_arg(bsplit, "output-buffer-duration", "1/100")
        dsp = Gst.ElementFactory.make("webrtcdsp")
        dsp.set_property("probe", self._AEC_PROBE_NAME)
        dsp.set_property("delay-agnostic", True)
        dsp.set_property("extended-filter", True)
        ac_out = Gst.ElementFactory.make("audioconvert")

        if not all([ac_in, capsf, bsplit, dsp, ac_out]):
            raise RuntimeError("Failed to create AEC DSP elements")

        for el in (ac_in, capsf, bsplit, dsp, ac_out):
            pipeline.add(el)
        src_element.link(ac_in)
        ac_in.link(capsf)
        capsf.link(bsplit)
        bsplit.link(dsp)
        dsp.link(ac_out)
        return ac_out

    def _attach_aec_probe(self, pipeline: Gst.Pipeline, tee: Gst.Element) -> None:
        """Attach the playback-side AEC reference branch to ``tee``.

        Adds a parallel branch: ``tee → queue → audioconvert →
        capsfilter(S16LE) → audiobuffersplit(10 ms) → webrtcechoprobe
        → fakesink``. The probe buffers reference frames internally
        and the recording-side ``webrtcdsp`` pulls from it via the
        shared name registered above.

        The terminal ``fakesink`` uses ``sync=False, async=False`` so
        the reference branch never throttles the rest of the playback
        pipeline (the speaker branch already paces it).
        """
        queue_p = Gst.ElementFactory.make("queue")
        ac_p = Gst.ElementFactory.make("audioconvert")
        capsf_p = self._make_aec_capsfilter()
        bsplit_p = Gst.ElementFactory.make("audiobuffersplit")
        # See note on the record-side bsplit: GstFraction setter
        # segfaults from Python; use the string-based helper instead.
        Gst.util_set_object_arg(bsplit_p, "output-buffer-duration", "1/100")
        probe = Gst.ElementFactory.make("webrtcechoprobe")
        probe.set_property("name", self._AEC_PROBE_NAME)
        fakesink_p = Gst.ElementFactory.make("fakesink")
        fakesink_p.set_property("sync", False)
        fakesink_p.set_property("async", False)
        fakesink_p.set_property("enable-last-sample", False)

        if not all([queue_p, ac_p, capsf_p, bsplit_p, probe, fakesink_p]):
            raise RuntimeError("Failed to create AEC probe elements")

        for el in (queue_p, ac_p, capsf_p, bsplit_p, probe, fakesink_p):
            pipeline.add(el)
        tee.link(queue_p)
        queue_p.link(ac_p)
        ac_p.link(capsf_p)
        capsf_p.link(bsplit_p)
        bsplit_p.link(probe)
        probe.link(fakesink_p)

    def _init_pipeline_record(self, pipeline: Gst.Pipeline) -> None:
        self._appsink_audio = Gst.ElementFactory.make("appsink")
        caps = Gst.Caps.from_string(
            f"audio/x-raw,rate={self.SAMPLE_RATE},channels={self.CHANNELS},format=F32LE,layout=interleaved"
        )
        self._appsink_audio.set_property("caps", caps)
        self._appsink_audio.set_property("drop", True)  # avoid overflow
        self._appsink_audio.set_property("max-buffers", 200)

        audiosrc: Optional[Gst.Element] = None

        if has_reachymini_asoundrc():
            # Wireless CM4: use the preconfigured .asoundrc ALSA devices
            # which route through the XMOS AEC loopback properly.
            audiosrc = Gst.ElementFactory.make("alsasrc")
            audiosrc.set_property("device", "reachymini_audio_src")
            self.logger.info("Using .asoundrc audio source: reachymini_audio_src")
        else:
            id_audio_card = get_audio_device("Source")

            if id_audio_card is None:
                self.logger.warning(
                    "No specific audio card found, using default audio source."
                )
                audiosrc = Gst.ElementFactory.make("autoaudiosrc")  # use default mic
            elif platform.system() == "Windows":
                audiosrc = Gst.ElementFactory.make("wasapi2src")
                audiosrc.set_property("device", id_audio_card)
            elif platform.system() == "Darwin":
                audiosrc = Gst.ElementFactory.make("osxaudiosrc")
                audiosrc.set_property("unique-id", id_audio_card)
            else:
                audiosrc = Gst.ElementFactory.make("pulsesrc")
                audiosrc.set_property("device", f"{id_audio_card}")

        queue = Gst.ElementFactory.make("queue")
        audioconvert = Gst.ElementFactory.make("audioconvert")
        audioresample = Gst.ElementFactory.make("audioresample")

        if not all([audiosrc, queue, audioconvert, audioresample, self._appsink_audio]):
            raise RuntimeError("Failed to create GStreamer elements")

        pipeline.add(audiosrc)
        pipeline.add(queue)
        pipeline.add(audioconvert)
        pipeline.add(audioresample)
        pipeline.add(self._appsink_audio)

        audiosrc.link(queue)
        queue.link(audioconvert)
        audioconvert.link(audioresample)
        # When no XVF3800 hardware is in the audio path, splice the
        # software AEC chain between `audioresample` and the appsink.
        # Otherwise link directly to keep the on-board AEC pristine.
        tail = (
            self._append_aec_dsp(pipeline, audioresample)
            if self._sw_aec_enabled
            else audioresample
        )
        tail.link(self._appsink_audio)

    def _build_audiosink_element(self) -> Gst.Element:
        """Create a platform-appropriate audio sink element."""
        audiosink: Optional[Gst.Element] = None

        if has_reachymini_asoundrc():
            audiosink = Gst.ElementFactory.make("alsasink")
            audiosink.set_property("device", "reachymini_audio_sink")
            self.logger.info("Using .asoundrc audio sink: reachymini_audio_sink")
        else:
            id_audio_card = get_audio_device("Sink")

            if id_audio_card is None:
                self.logger.warning(
                    "No specific audio card found, using default audio sink."
                )
                audiosink = Gst.ElementFactory.make("autoaudiosink")
            elif platform.system() == "Windows":
                audiosink = Gst.ElementFactory.make("wasapi2sink")
                audiosink.set_property("device", id_audio_card)
            elif platform.system() == "Darwin":
                audiosink = Gst.ElementFactory.make("osxaudiosink")
                audiosink.set_property("unique-id", id_audio_card)
            else:
                audiosink = Gst.ElementFactory.make("pulsesink")
                audiosink.set_property("device", f"{id_audio_card}")

        if audiosink is None:
            raise RuntimeError("Failed to create audio sink element")

        if audiosink.find_property("buffer-time") is not None:
            audiosink.set_property("buffer-time", self.PLAYBACK_SINK_BUFFER_TIME_US)
        if audiosink.find_property("latency-time") is not None:
            audiosink.set_property("latency-time", self.PLAYBACK_SINK_LATENCY_TIME_US)

        return audiosink

    def _make_wobbler_appsink(self) -> Gst.Element:
        """Create an appsink that feeds audio to the head wobbler.

        ``sync=True`` so new-sample fires at the buffer's PTS on the
        pipeline clock — i.e. when the audiosink outputs it. The local
        pipeline has a deterministic clock and no network jitter, so
        PTS-based sync gives correct A/V timing for both playbin
        (``play_sound``) and push (``push_audio_sample``) paths.
        """
        appsink = Gst.ElementFactory.make("appsink")
        # Force mono so the speech tapper receives a 1-D float32 array.
        # The per-branch audioconvert in _build_audiosink_tee_bin /
        # _init_pipeline_playback handles the downmix.
        caps = Gst.Caps.from_string(
            f"audio/x-raw,format=F32LE,channels=1,"
            f"rate={self.SAMPLE_RATE},layout=interleaved"
        )
        appsink.set_property("caps", caps)
        appsink.set_property("drop", True)
        appsink.set_property("max-buffers", 5)
        appsink.set_property("sync", True)
        appsink.set_property("emit-signals", True)
        appsink.connect("new-sample", self._on_wobbler_sample)
        return appsink

    def _on_wobbler_sample(self, appsink: Gst.Element) -> Gst.FlowReturn:
        """GStreamer callback: forward audio buffer to the head wobbler.

        The appsink is ``sync=True``, so this callback fires at the
        buffer's PTS on the pipeline clock — audio is playing NOW.
        """
        sample = appsink.pull_sample()
        if sample is None or self._head_wobbler is None:
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        data = buf.extract_dup(0, buf.get_size())
        pcm = np.frombuffer(data, dtype=np.float32)
        self._head_wobbler.feed(pcm, time.monotonic_ns())
        return Gst.FlowReturn.OK

    def _build_audiosink_tee_bin(self) -> Gst.Bin:
        """Build a Gst.Bin with a tee splitting audio to speaker and wobbler.

        Per-branch audioconvert+audioresample isolate each leaf's caps
        from the other (the wobbler appsink demands F32LE/2/16000; the
        audiosink wants whatever the device prefers — e.g. on the
        wireless XMOS PCM, anything but its native rate triggers an
        IEC958 fallback that fails to open).

        The bin exposes a single ghost sink pad for use as a playbin audio-sink::

            ghost_sink → tee ─┬→ queue → audioconvert → audioresample → audiosink
                               └→ queue → audioconvert → audioresample → appsink

        """
        audio_bin = Gst.Bin.new("audio_tee_bin")

        tee = Gst.ElementFactory.make("tee")
        queue_speaker = Gst.ElementFactory.make("queue")
        ac_speaker = Gst.ElementFactory.make("audioconvert")
        ar_speaker = Gst.ElementFactory.make("audioresample")
        audiosink = self._build_audiosink_element()
        queue_wobbler = Gst.ElementFactory.make("queue")
        ac_wobbler = Gst.ElementFactory.make("audioconvert")
        ar_wobbler = Gst.ElementFactory.make("audioresample")
        appsink_wobbler = self._make_wobbler_appsink()

        for el in (
            tee,
            queue_speaker,
            ac_speaker,
            ar_speaker,
            audiosink,
            queue_wobbler,
            ac_wobbler,
            ar_wobbler,
            appsink_wobbler,
        ):
            audio_bin.add(el)

        tee.link(queue_speaker)
        queue_speaker.link(ac_speaker)
        ac_speaker.link(ar_speaker)
        ar_speaker.link(audiosink)

        tee.link(queue_wobbler)
        queue_wobbler.link(ac_wobbler)
        ac_wobbler.link(ar_wobbler)
        ar_wobbler.link(appsink_wobbler)

        ghost_pad = Gst.GhostPad.new("sink", tee.get_static_pad("sink"))
        audio_bin.add_pad(ghost_pad)

        return audio_bin

    def _init_pipeline_playback(self, pipeline: Gst.Pipeline) -> None:
        self._appsrc = Gst.ElementFactory.make("appsrc")
        self._appsrc.set_property("do-timestamp", False)
        self._appsrc.set_property("format", Gst.Format.TIME)
        self._appsrc.set_property("is-live", True)
        caps = Gst.Caps.from_string(
            f"audio/x-raw,format=F32LE,channels={self.CHANNELS},rate={self.SAMPLE_RATE},layout=interleaved"
        )
        self._appsrc.set_property("caps", caps)

        # Always build tee so wobbling can be enabled/disabled at runtime.
        # Per-branch audioconvert+audioresample so the wobbler appsink's
        # F32LE/1/16000 caps don't drag the audiosink branch into a rate
        # the device can't accept (e.g. wireless XMOS PCM falls back to
        # IEC958 at non-native rates). The appsink with drop=True has
        # negligible overhead when no wobbler is connected.
        tee = Gst.ElementFactory.make("tee")
        queue_speaker = Gst.ElementFactory.make("queue")
        ac_speaker = Gst.ElementFactory.make("audioconvert")
        ar_speaker = Gst.ElementFactory.make("audioresample")
        audiosink = self._build_audiosink_element()
        queue_wobbler = Gst.ElementFactory.make("queue")
        ac_wobbler = Gst.ElementFactory.make("audioconvert")
        ar_wobbler = Gst.ElementFactory.make("audioresample")
        appsink_wobbler = self._make_wobbler_appsink()

        for el in (
            self._appsrc,
            tee,
            queue_speaker,
            ac_speaker,
            ar_speaker,
            audiosink,
            queue_wobbler,
            ac_wobbler,
            ar_wobbler,
            appsink_wobbler,
        ):
            pipeline.add(el)

        self._appsrc.link(tee)
        tee.link(queue_speaker)
        queue_speaker.link(ac_speaker)
        ac_speaker.link(ar_speaker)
        ar_speaker.link(audiosink)
        tee.link(queue_wobbler)
        queue_wobbler.link(ac_wobbler)
        ac_wobbler.link(ar_wobbler)
        ar_wobbler.link(appsink_wobbler)

        # Software AEC reference branch. We only attach it to the
        # appsrc-based playback pipeline (i.e. conversation audio
        # pushed via `push_audio_sample`), NOT to the `playbin`-driven
        # tee bin used by `play_sound`. Two reasons:
        #   1. Sound-file beeps are short and not interpretable by an
        #      LLM as speech even if they leak into the mic, so the
        #      cost/benefit doesn't justify the extra wiring.
        #   2. `webrtcechoprobe` registers a process-wide name, so
        #      keeping it bound to a single, long-lived element is the
        #      simplest way to avoid duplicate-name collisions when a
        #      playbin tee bin spins up mid-conversation.
        if self._sw_aec_enabled:
            self._attach_aec_probe(pipeline, tee)

    def _on_bus_message(
        self, bus: Gst.Bus, msg: Gst.Message, pipeline: Gst.Pipeline
    ) -> bool:
        if msg.type == Gst.MessageType.EOS and self._head_wobbler is not None:
            self._head_wobbler.stop()
        return super()._on_bus_message(bus, msg, pipeline)

    def _dump_latency(self) -> None:
        query = Gst.Query.new_latency()
        self._pipeline_playback.query(query)
        self.logger.info(f"Audio pipeline latency {query.parse_latency()}")

    def start_recording(self) -> None:
        """Start capturing audio from the microphone."""
        self._pipeline_record.set_state(Gst.State.PLAYING)

    def stop_recording(self) -> None:
        """Stop the recording pipeline."""
        self._pipeline_record.set_state(Gst.State.NULL)

    def start_playing(self) -> None:
        """Start the playback pipeline so ``push_audio_sample`` can feed data."""
        if self._head_wobbler is not None:
            self._head_wobbler.start()
        self._appsrc_pts = -1
        self._pipeline_playback.set_state(Gst.State.PLAYING)
        GLib.timeout_add_seconds(5, self._dump_latency)

    def push_audio_sample(self, data: npt.NDArray[np.float32]) -> None:
        """Push audio data to the speaker.

        Args:
            data: Audio samples as a float32 array.  Shape should be
                ``(num_samples, 2)`` for stereo or ``(num_samples,)`` for
                mono (the caller is responsible for channel adaptation).

        """
        if self._appsrc is None:
            self.logger.warning(
                "AppSrc is not initialized. Call start_playing() first."
            )
            return

        pts_ns, duration_ns, self._appsrc_pts = self._compute_pts(
            int(data.shape[0]),
            self._appsrc.get_current_running_time(),
            self._appsrc_pts,
        )
        buf = Gst.Buffer.new_wrapped(data.tobytes())
        buf.pts = pts_ns
        buf.dts = pts_ns
        buf.duration = duration_ns
        ret = self._appsrc.push_buffer(buf)
        if ret != Gst.FlowReturn.OK:
            self.logger.warning(f"push_buffer dropped: {ret}")

    def stop_playing(self) -> None:
        """Stop the playback pipeline."""
        if self._head_wobbler is not None:
            self._head_wobbler.stop()
        self._appsrc_pts = -1
        self._pipeline_playback.set_state(Gst.State.NULL)
        if self._playbin is not None:
            self._playbin.set_state(Gst.State.NULL)
            self._playbin = None

    def clear_output_buffer(self) -> None:
        """Flush queued playback data so it is not played.

        A low ``set_max_output_buffers`` value may make this unnecessary
        for most use-cases.

        """
        pass  # subclasses or future implementations can override

    def clear_player(self) -> None:
        """Flush the player's appsrc to drop any queued audio immediately."""
        if self._head_wobbler is not None:
            self._head_wobbler.reset()
        if self._appsrc is not None:
            self._appsrc_pts = -1
            self._pipeline_playback.set_state(Gst.State.PAUSED)
            self._appsrc.send_event(Gst.Event.new_flush_start())
            self._appsrc.send_event(Gst.Event.new_flush_stop(reset_time=True))
            self._pipeline_playback.set_state(Gst.State.PLAYING)
            self.logger.info("Cleared player queue")
        else:
            self.logger.warning(
                "AppSrc is not initialized. Call start_playing() first."
            )

    def play_sound(self, sound_file: str) -> None:
        """Play a sound file through the Reachy Mini Audio card.

        The file is played via a GStreamer ``playbin`` routed to the same
        audio sink used by the push-based playback pipeline.  When the head
        wobbler is enabled the audio is also forked to it via a tee.

        Args:
            sound_file: Absolute path **or** filename relative to the
                built-in assets directory.

        Raises:
            FileNotFoundError: If the file cannot be found.

        """
        if not os.path.exists(sound_file):
            file_path = f"{ASSETS_ROOT_PATH}/{sound_file}"
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Sound file {sound_file} not found in assets directory or given path."
                )
        else:
            file_path = sound_file

        if self._playbin is not None:
            self._playbin.set_state(Gst.State.NULL)

        playbin = Gst.ElementFactory.make("playbin", "player")
        if not playbin:
            self.logger.error("Failed to create playbin element")
            return

        # Fix for Windows: use file:/// and forward slashes
        if os.name == "nt":
            uri_path = file_path.replace("\\", "/")
            if not uri_path.startswith("/") and ":" in uri_path:
                # Ensure three slashes after file: for absolute paths (file:///C:/...)
                uri = f"file:///{uri_path}"
            else:
                uri = f"file://{uri_path}"
        else:
            uri = f"file://{file_path}"
        playbin.set_property("uri", uri)

        playbin.set_property("audio-sink", self._build_audiosink_tee_bin())
        if self._head_wobbler is not None:
            self._head_wobbler.reset()
            self._head_wobbler.start()

        self._playbin = playbin
        playbin.set_state(Gst.State.PLAYING)

    def upload_sound(self, sound_file: str) -> str:
        """No-op for the local backend — the file is already accessible.

        Returns:
            The unchanged *sound_file* path.

        """
        return sound_file

    def list_sounds(self) -> list[str]:
        """No-op for the local backend.

        Returns:
            An empty list.

        """
        return []

    def delete_sound(self, filename: str) -> bool:
        """No-op for the local backend.

        Returns:
            Always ``False``.

        """
        return False

    def get_DoA(self) -> tuple[float, bool] | None:
        """Get the Direction of Arrival (DoA) from the ReSpeaker.

        Returns:
            A tuple ``(angle_radians, speech_detected)`` or ``None``
            if the device is unavailable.

        """
        return self._doa.get_DoA()

    def enable_wobbling(self, callback: Callable[[SpeechOffsets], None]) -> None:
        """Enable head wobbling driven by audio playback.

        Args:
            callback: Called with ``(x_m, y_m, z_m, roll_rad, pitch_rad,
                yaw_rad)`` for each movement hop.

        """
        if self._head_wobbler is not None:
            self._head_wobbler.stop()
        self._head_wobbler = HeadWobbler(callback, sample_rate=self.SAMPLE_RATE)
        self.logger.info("Head wobbler enabled")

    def disable_wobbling(self) -> None:
        """Disable head wobbling."""
        if self._head_wobbler is not None:
            self._head_wobbler.stop()
            self._head_wobbler = None
            self.logger.info("Head wobbler disabled")

    def cleanup(self) -> None:
        """Release all resources (pipelines, USB devices)."""
        if self._head_wobbler is not None:
            self._head_wobbler.stop()
        self._doa.close()

    def __del__(self) -> None:
        """Ensure GStreamer resources are released."""
        self.cleanup()
        self._loop.quit()
        self._bus_record.remove_watch()
        self._bus_playback.remove_watch()
