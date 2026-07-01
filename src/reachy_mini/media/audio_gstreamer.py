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

from reachy_mini.media.audio_base import AEC_PROBE_NAME, AEC_RATE, AudioBase
from reachy_mini.media.audio_utils import has_reachymini_asoundrc
from reachy_mini.media.device_detection import DEFAULT_AUDIO_TARGET, get_audio_device
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

    Extends ``AudioBase`` with a GStreamer-specific helper:

    - ``clear_player()``: flush the playback appsrc immediately via GStreamer
      flush events, dropping any queued audio.

    (``clear_output_buffer()`` is deprecated and does nothing; use
    ``clear_player()`` instead.)

    """

    PLAYBACK_SINK_BUFFER_TIME_US = 50_000
    PLAYBACK_SINK_LATENCY_TIME_US = 5_000

    def __init__(
        self,
        input_device: Optional[str] = None,
        output_device: Optional[str] = None,
        log_level: str = "INFO",
    ) -> None:
        """Initialize recording and playback pipelines.

        Args:
            input_device: Input device (mic) name, or ``None`` to keep the
                auto-detected source.
            output_device: Output device (speaker) name, or ``None`` to keep
                the auto-detected sink.
            log_level: Logging level for audio operations.
                Options: ``'DEBUG'``, ``'INFO'``, ``'WARNING'``, ``'ERROR'``,
                ``'CRITICAL'``.

        """
        super().__init__(log_level=log_level)

        # User-selected device overrides. Read by _init_pipeline_record and
        # _build_audiosink_element, so they must be set before those run.
        self._input_device = input_device
        self._output_device = output_device

        self._head_wobbler: Optional[HeadWobbler] = None

        Gst.init([])
        self._loop = GLib.MainLoop()
        self._thread_bus_calls = Thread(target=lambda: self._loop.run(), daemon=True)
        self._thread_bus_calls.start()

        self._webrtcechoprobe: Optional[Gst.Element] = None

        # Single pipeline holds both record and playback chains so that
        # webrtcdsp + webrtcechoprobe share a clock (required by the
        # GStreamer webrtcdsp docs to align the far-end reference with
        # the mic capture).
        self._pipeline = Gst.Pipeline.new("reachymini_audio")
        self._init_pipeline_record(self._pipeline)
        self._bus = self._pipeline.get_bus()
        self._bus.add_watch(GLib.PRIORITY_DEFAULT, self._on_bus_message, self._pipeline)

        self._playbin: Optional[Gst.Element] = None
        self._init_pipeline_playback(self._pipeline)

    def _init_pipeline_record(self, pipeline: Gst.Pipeline) -> None:
        self._appsink_audio = Gst.ElementFactory.make("appsink")
        caps = Gst.Caps.from_string(
            f"audio/x-raw,rate={self.SAMPLE_RATE},channels={self.CHANNELS},format=F32LE,layout=interleaved"
        )
        self._appsink_audio.set_property("caps", caps)
        self._appsink_audio.set_property("drop", True)  # avoid overflow
        self._appsink_audio.set_property("max-buffers", 200)

        audiosrc: Optional[Gst.Element] = None
        webrtcdsp: Optional[Gst.Element] = None

        # An explicit user selection overrides the .asoundrc / auto-detected
        # source. It is resolved platform-appropriately by get_audio_device (via
        # its target_name), so the right element is used on every OS. AEC is
        # only wired on the default-fallback path (no specific card found).
        if self._input_device is None and has_reachymini_asoundrc():
            # Wireless CM4: use the preconfigured .asoundrc ALSA devices
            # which route through the XMOS AEC loopback properly.
            audiosrc = Gst.ElementFactory.make("alsasrc")
            audiosrc.set_property("device", "reachymini_audio_src")
            self.logger.info("Using .asoundrc audio source: reachymini_audio_src")
        else:
            id_audio_card = get_audio_device(
                "Source", target_name=self._input_device or DEFAULT_AUDIO_TARGET
            )

            if id_audio_card is None:
                self.logger.warning(
                    "No specific audio card found, using default audio source."
                )
                audiosrc = Gst.ElementFactory.make("autoaudiosrc")  # use default mic
                self._webrtcechoprobe = Gst.ElementFactory.make("webrtcechoprobe")
                webrtcdsp = Gst.ElementFactory.make("webrtcdsp")
                if self._webrtcechoprobe is None or webrtcdsp is None:
                    self.logger.warning(
                        "Cannot enable webrtcdsp. Check if gst-plugins-bad are available."
                    )
                    # Drop both so the playback chain doesn't wire a probe
                    # without a matching DSP (or vice versa).
                    self._webrtcechoprobe = None
                    webrtcdsp = None
                else:
                    # Pair probe ↔ dsp so the playback signal is used as the
                    # far-end reference for echo cancellation on the mic path.
                    self._webrtcechoprobe.set_property("name", AEC_PROBE_NAME)
                    webrtcdsp.set_property("probe", AEC_PROBE_NAME)
                    self.logger.info("Enabling webRTC echo cancellation.")
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

        if webrtcdsp:
            # webrtcdsp requires S16LE at 8/16/32/48 kHz — convert/resample
            # in, then convert back to F32LE at SAMPLE_RATE for the appsink.
            ac_in = Gst.ElementFactory.make("audioconvert")
            ar_in = Gst.ElementFactory.make("audioresample")
            cf_in = Gst.ElementFactory.make("capsfilter")
            cf_in.set_property(
                "caps",
                Gst.Caps.from_string(
                    f"audio/x-raw,format=S16LE,rate={AEC_RATE},"
                    f"channels={self.CHANNELS},layout=interleaved"
                ),
            )
            for el in (ac_in, ar_in, cf_in, webrtcdsp):
                pipeline.add(el)
            audiosrc.link(ac_in)
            ac_in.link(ar_in)
            ar_in.link(cf_in)
            cf_in.link(webrtcdsp)
            webrtcdsp.link(queue)
        else:
            audiosrc.link(queue)

        queue.link(audioconvert)
        audioconvert.link(audioresample)
        audioresample.link(self._appsink_audio)

    def _build_audiosink_element(self) -> Gst.Element:
        """Create a platform-appropriate audio sink element."""
        audiosink: Optional[Gst.Element] = None

        # An explicit user selection overrides the .asoundrc / auto-detected
        # sink. It is resolved platform-appropriately by get_audio_device (via
        # its target_name), so the right element is used on every OS.
        if self._output_device is None and has_reachymini_asoundrc():
            audiosink = Gst.ElementFactory.make("alsasink")
            audiosink.set_property("device", "reachymini_audio_sink")
            self.logger.info("Using .asoundrc audio sink: reachymini_audio_sink")
        else:
            id_audio_card = get_audio_device(
                "Sink", target_name=self._output_device or DEFAULT_AUDIO_TARGET
            )

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
        # We stamp the first buffer of each utterance ourselves (DISCONT +
        # running-time PTS) and leave follow-up buffers with no timestamp
        # so the audiomixer places them contiguously by byte offset.
        self._appsrc.set_property("do-timestamp", False)
        self._appsrc.set_property("format", Gst.Format.TIME)
        self._appsrc.set_property("is-live", True)
        caps = Gst.Caps.from_string(
            f"audio/x-raw,format=F32LE,channels={self.CHANNELS},rate={self.SAMPLE_RATE},layout=interleaved"
        )
        self._appsrc.set_property("caps", caps)

        mixer = Gst.ElementFactory.make("audiomixer")
        appsrc_queue = Gst.ElementFactory.make("queue")

        silence = Gst.ElementFactory.make("audiotestsrc")
        silence.set_property("is-live", True)
        silence.set_property("wave", 4)  # silence
        # Pin silence caps to the appsrc format so audiomixer sees matching
        # inputs (it can't aggregate F32LE@N and S16@44100 together).
        silence_caps = Gst.ElementFactory.make("capsfilter")
        silence_caps.set_property(
            "caps",
            Gst.Caps.from_string(
                f"audio/x-raw,format=F32LE,channels={self.CHANNELS},"
                f"rate={self.SAMPLE_RATE},layout=interleaved"
            ),
        )
        silence_queue = Gst.ElementFactory.make("queue")

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
            appsrc_queue,
            silence,
            silence_caps,
            silence_queue,
            mixer,
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

        self._appsrc.link(appsrc_queue)
        appsrc_queue.link(mixer)
        silence.link(silence_caps)
        silence_caps.link(silence_queue)
        silence_queue.link(mixer)

        if self._webrtcechoprobe is not None:
            # webrtcechoprobe requires S16LE at 8/16/32/48 kHz.
            ac_probe = Gst.ElementFactory.make("audioconvert")
            ar_probe = Gst.ElementFactory.make("audioresample")
            cf_probe = Gst.ElementFactory.make("capsfilter")
            cf_probe.set_property(
                "caps",
                Gst.Caps.from_string(
                    f"audio/x-raw,format=S16LE,rate={AEC_RATE},"
                    f"channels={self.CHANNELS},layout=interleaved"
                ),
            )
            for el in (ac_probe, ar_probe, cf_probe, self._webrtcechoprobe):
                pipeline.add(el)
            mixer.link(ac_probe)
            ac_probe.link(ar_probe)
            ar_probe.link(cf_probe)
            cf_probe.link(self._webrtcechoprobe)
            self._webrtcechoprobe.link(tee)
        else:
            mixer.link(tee)

        tee.link(queue_speaker)
        queue_speaker.link(ac_speaker)
        ac_speaker.link(ar_speaker)
        ar_speaker.link(audiosink)
        tee.link(queue_wobbler)
        queue_wobbler.link(ac_wobbler)
        ac_wobbler.link(ar_wobbler)
        ar_wobbler.link(appsink_wobbler)

    def _on_bus_message(
        self, bus: Gst.Bus, msg: Gst.Message, pipeline: Gst.Pipeline
    ) -> bool:
        if msg.type == Gst.MessageType.EOS and self._head_wobbler is not None:
            self._head_wobbler.stop()
        return super()._on_bus_message(bus, msg, pipeline)

    def _dump_latency(self) -> None:
        query = Gst.Query.new_latency()
        self._pipeline.query(query)
        self.logger.info(f"Audio pipeline latency {query.parse_latency()}")

    def start_recording(self) -> None:
        """Start capturing audio from the microphone."""
        self._pipeline.set_state(Gst.State.PLAYING)

    def stop_recording(self) -> None:
        """Stop the recording pipeline."""
        self._pipeline.set_state(Gst.State.NULL)

    def start_playing(self) -> None:
        """Start the playback pipeline so ``push_audio_sample`` can feed data."""
        if self._head_wobbler is not None:
            self._head_wobbler.start()
        self._appsrc_pts = -1
        self._pipeline.set_state(Gst.State.PLAYING)
        GLib.timeout_add_seconds(5, self._dump_latency)

    def stop_playing(self) -> None:
        """Stop the playback pipeline."""
        if self._head_wobbler is not None:
            self._head_wobbler.stop()
        self._appsrc_pts = -1
        self._pipeline.set_state(Gst.State.NULL)
        if self._playbin is not None:
            self._playbin.set_state(Gst.State.NULL)
            self._playbin = None

    def clear_player(self) -> None:
        """Flush the player's appsrc to drop any queued audio immediately."""
        if self._head_wobbler is not None:
            self._head_wobbler.reset()
        if self._appsrc is not None:
            self._appsrc_pts = -1
            self._pipeline.set_state(Gst.State.PAUSED)
            self._appsrc.send_event(Gst.Event.new_flush_start())
            self._appsrc.send_event(Gst.Event.new_flush_stop(reset_time=True))
            self._pipeline.set_state(Gst.State.PLAYING)
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
        self._bus.remove_watch()
