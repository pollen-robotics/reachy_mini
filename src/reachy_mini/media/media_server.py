"""GStreamer media server for the Reachy Mini daemon.

Owns the physical camera and audio hardware and distributes media to
consumers through two channels:

- **WebRTC** — streams video + audio to remote clients (browsers, remote
  Python SDK) via ``webrtcsink`` and a signalling server.
- **IPC** — shares raw BGR frames with on-device applications via
  ``unixfdsink`` (Linux / macOS) or ``win32ipcvideosink`` (Windows),
  avoiding encode / decode overhead.

The server is started by the Reachy Mini daemon on **all** platforms
(Lite and Wireless).  It also provides a ``play_sound()`` method for
playing sound files directly on the robot's speaker.

Example usage::

    >>> from reachy_mini.media.media_server import GstMediaServer
    >>>
    >>> server = GstMediaServer(log_level="INFO")
    >>> server.start()
    >>> # The server is now streaming and ready to accept client connections
"""

import logging
import os
import platform
import time
from dataclasses import dataclass, field
from threading import Lock, Thread
from typing import Any, Callable, Dict, Optional

import gi
import numpy as np

from reachy_mini.daemon.utils import (
    CAMERA_PIPE_NAME,
    CAMERA_SOCKET_PATH,
    SimulationMode,
    is_local_camera_available,
)
from reachy_mini.media.audio_base import AEC_CHANNELS, AEC_PROBE_NAME, AEC_RATE
from reachy_mini.media.audio_control_utils import init_respeaker_usb
from reachy_mini.media.audio_utils import has_reachymini_asoundrc
from reachy_mini.media.camera_constants import (
    CameraSpecs,
    GenericWebcamSpecs,
    MujocoCameraSpecs,
    ReachyMiniLiteCamSpecs,
)
from reachy_mini.media.device_detection import (
    DEFAULT_AUDIO_TARGET,
    get_audio_device,
    get_video_device,
)
from reachy_mini.media.gstreamer_utils import handle_default_bus_message
from reachy_mini.motion.head_wobbler import HeadWobbler, SpeechOffsets
from reachy_mini.utils.constants import ASSETS_ROOT_PATH

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")

from gi.repository import GLib, Gst, GstApp  # noqa: E402, F401

# Hard cap on how long a freshly-added consumer is allowed to spend
# before its `webrtcbin.connection-state` reaches "connected". In a
# healthy run the negotiation completes in well under a second
# (offer/answer + a few ICE candidate pairs). Past this deadline,
# we conclude the negotiation is stuck — the typical culprit is
# libnice frozen mid-`CHECKING` (a known crash mode of certain
# libnice versions) or a downstream networking issue we can't see
# from the daemon. We notify the central so the JS client gets a
# clean rejection instead of a spinner-on-blank-page UX.
#
# 12 s is generous for a healthy negotiation and short enough that
# the user sees the failure quickly. Tuned conservatively because
# the daemon has no way to distinguish "slow ICE on a flaky
# network" from "libnice frozen" — at this duration both are bad.
ICE_NEGOTIATION_DEADLINE_S = 12

# `reason` strings sent to central via the existing `endSession`
# message. Picked to be parseable by the JS SDK so a user-facing
# message can map to a precise cause. See
# `central_signaling_relay.notify_peer_session_failed` for the
# wire-level emission.
SESSION_FAILED_REASON_ICE_TIMEOUT = "ice_negotiation_timeout"
SESSION_FAILED_REASON_PC_FAILED = "peer_connection_failed"


@dataclass
class _PeerWebRTCState:
    """Live state of a single WebRTC peer's negotiation.

    Used by the watchdog to decide if the session is making progress.
    Updated on every `notify::*-state` callback fired by webrtcbin.
    Held under :attr:`GstMediaServer._peer_states_lock` because
    GStreamer signals can fire on internal threads.
    """

    peer_id: str
    ice_state: str = "new"
    conn_state: str = "new"
    signaling_state: str = "new"
    added_at: float = field(default_factory=time.monotonic)
    # GLib timer source ID (returned by `GLib.timeout_add_seconds`).
    # Kept so the watchdog can be cancelled if the peer reaches a
    # terminal state (connected, removed) before the deadline.
    watchdog_source_id: Optional[int] = None
    # Whether the failure callback has already fired for this peer.
    # Prevents double-notification when both `connection-state ==
    # failed` AND the deadline timer fire close together.
    failure_notified: bool = False

    def asdict(self) -> Dict[str, Any]:
        """Build a serialisable snapshot for diagnostics in failure notifications."""
        return {
            "peer_id": self.peer_id,
            "ice_state": self.ice_state,
            "conn_state": self.conn_state,
            "signaling_state": self.signaling_state,
            "elapsed_s": round(time.monotonic() - self.added_at, 2),
        }


class GstMediaServer:
    """Daemon-side GStreamer media server.

    Owns the camera and audio hardware and distributes media to consumers:

    - **IPC branch** — raw BGR frames via ``unixfdsink`` / ``win32ipcvideosink``
      for on-device applications (``GStreamerCamera`` reads from this).
    - **WebRTC branch** — encoded video + audio via ``webrtcsink`` for remote
      clients (``GstWebRTCClient`` connects to this).
    - **Sound playback** — ``playbin`` for playing WAV files on the speaker.

    Attributes:
        camera_specs (CameraSpecs): Specifications of the detected camera.
        resized_K (npt.NDArray[np.float64]): Camera intrinsic matrix for current resolution.

    """

    # Sample rate the wobbler appsink demands; the per-branch audioresample
    # converts whatever the source produces down to this rate before delivery.
    WOBBLER_SAMPLE_RATE = 16_000

    # Receive-side jitter buffer depth (ms) for the consumer `webrtcbin`,
    # i.e. the phone->robot voice leg. Default webrtcbin latency (200ms)
    # underruns on a jittery Wi-Fi link and the speaker stutters. We trade
    # a little latency for a steady buffer, capped at 300ms because
    # end-to-end latency is the metric we watch closest.
    RX_JITTER_LATENCY_MS = 300

    # Send-side Opus loss resilience for the robot mic -> phone leg (the
    # audio that feeds the realtime backend / STT). webrtcsink builds the
    # opusenc with defaults (inband-fec off, packet-loss-percentage 0), so
    # a dropped mic packet on a jittery Wi-Fi link can only be concealed by
    # the browser decoder, never reconstructed. We enable in-band FEC and
    # tell the encoder to budget for this much loss so it ships redundancy.
    TX_OPUS_FEC_LOSS_PERC = 20

    # Name of the appsrc feeding the incoming-audio playback pipeline; used
    # both when building the pipeline and when flushing it (clear_incoming_audio).
    INCOMING_AUDIO_SRC_NAME = "audio_in"

    def __init__(
        self,
        log_level: str = "INFO",
        sim_mode: SimulationMode = SimulationMode.NONE,
    ) -> None:
        """Initialize the GStreamer WebRTC pipeline.

        Args:
            log_level: Logging level for WebRTC daemon operations.
            sim_mode: Simulation mode. MUJOCO receives video via UDP,
                MOCKUP uses autovideosrc, NONE detects a physical camera.

        Raises:
            RuntimeError: If no camera is detected (unless in simulation mode)
                or camera specifications cannot be determined.

        """
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)
        self._log_level = log_level
        self._sim_mode = sim_mode

        Gst.init([])
        self._loop = GLib.MainLoop()
        self._thread_bus_calls = Thread(target=lambda: self._loop.run(), daemon=True)
        self._thread_bus_calls.start()

        match sim_mode:
            case SimulationMode.MUJOCO:
                cam_path = "use_sim"
                self.camera_specs: CameraSpecs = MujocoCameraSpecs()
            case SimulationMode.MOCKUP:
                cam_path = "use_mockup_sim"
                self.camera_specs = GenericWebcamSpecs()
            case SimulationMode.NONE:
                cam_path, detected_specs = get_video_device()
                if detected_specs is None:
                    self._logger.warning(
                        "No camera found. Video will not be available."
                    )
                    self.camera_specs = ReachyMiniLiteCamSpecs()
                else:
                    self.camera_specs = detected_specs

        self._resolution = self.camera_specs.default_resolution
        self.resized_K = self.camera_specs.K

        if self._resolution is None:
            raise RuntimeError("Failed to get default camera resolution.")

        self._cam_path = cam_path

        self._data_channels: dict[str, Gst.Element] = {}  # peer_id -> channel
        self._on_data_message: Optional[Callable[[str, str], None]] = None
        # Optional callback fired on the GStreamer thread when a peer
        # leaves; used by the backend to free per-peer resources such
        # as the journalctl subprocess for a `subscribe_logs` stream.
        self._on_peer_disconnect: Optional[Callable[[str], None]] = None
        # Optional callback fired by the ICE negotiation watchdog when
        # a peer's `webrtcbin` is stuck mid-negotiation (see
        # `_check_negotiation_deadline`). Wired by the daemon to the
        # central signaling relay so the JS client gets a typed
        # `endSession` instead of a spinner. Signature:
        # `(peer_id, reason, diagnostic_dict) -> None`.
        self._on_session_failed: Optional[
            Callable[[str, str, Dict[str, Any]], None]
        ] = None
        self._peer_states: Dict[str, _PeerWebRTCState] = {}
        # GStreamer signals (`notify::*`) can fire on internal threads
        # owned by webrtcbin / libnice, while consumer-added /
        # consumer-removed run on the GLib main thread. The state
        # dict is touched from both, so we take a lock around every
        # mutation. The critical sections are tiny (a few field
        # writes) so contention is negligible.
        self._peer_states_lock = Lock()
        self._incoming_audio: Dict[str, Dict[str, Any]] = {}
        self._playbin: Optional[Gst.Element] = None
        self._head_wobbler: Optional[HeadWobbler] = None
        self._pipeline_playback: Optional[Gst.Pipeline] = None
        # Software AEC: set in _configure_audio; the probe is created there and
        # reused across per-peer playback pipelines (see _on_consumer_pad_added).
        self._aec_enabled = False
        self._webrtcechoprobe: Optional[Gst.Element] = None

        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Build (or rebuild) the GStreamer pipeline from scratch."""
        self._pipeline_sender = Gst.Pipeline.new("reachymini_webrtc_sender")
        self._bus_sender = self._pipeline_sender.get_bus()
        self._bus_sender.add_watch(
            GLib.PRIORITY_DEFAULT, self._on_bus_message, self._pipeline_sender
        )

        webrtcsink = self._configure_webrtc(self._pipeline_sender)

        self._configure_video(self._cam_path, self._pipeline_sender, webrtcsink)
        self._configure_audio(self._pipeline_sender, webrtcsink)

        self._logger.debug("Pipeline built")

    def close(self) -> None:
        """Release GStreamer resources (MainLoop, bus watch)."""
        self._logger.debug("Cleaning up GstMediaServer")
        self._loop.quit()
        self._bus_sender.remove_watch()

    def __del__(self) -> None:
        """Destructor to ensure gstreamer resources are released."""
        self.close()

    def _dump_latency(self) -> None:
        query = Gst.Query.new_latency()
        self._pipeline_sender.query(query)
        self._logger.info(f"Pipeline latency {query.parse_latency()}")

    def _configure_webrtc(self, pipeline: Gst.Pipeline) -> Gst.Element:
        self._logger.debug("Configuring WebRTC")
        webrtcsink = Gst.ElementFactory.make("webrtcsink")
        if not webrtcsink:
            raise RuntimeError(
                "Failed to create webrtcsink element. "
                "Is the GStreamer webrtc rust plugin installed?"
            )

        meta_structure = Gst.Structure.new_empty("meta")
        meta_structure.set_value("name", "reachymini")
        webrtcsink.set_property("meta", meta_structure)
        webrtcsink.set_property("run-signalling-server", True)

        webrtcsink.connect("consumer-added", self._consumer_added)
        webrtcsink.connect("consumer-removed", self._consumer_removed)
        # Tune the auto-created Opus encoder for the mic->phone leg
        # (in-band FEC). See `_encoder_setup` / `TX_OPUS_FEC_LOSS_PERC`.
        webrtcsink.connect("encoder-setup", self._encoder_setup)

        pipeline.add(webrtcsink)

        return webrtcsink

    def _encoder_setup(
        self,
        webrtcsink: Gst.Element,
        consumer_id: str,
        pad_name: str,
        encoder: Gst.Element,
    ) -> bool:
        """Configure webrtcsink's auto-created encoder before it runs.

        Fired by ``webrtcsink`` once per consumer encoder. We only touch
        the Opus audio encoder (the robot mic uplink): enable in-band FEC
        and budget for packet loss so the encoder ships redundancy that
        the browser can use to reconstruct dropped mic packets instead of
        merely concealing them. Returning ``False`` keeps webrtcsink's own
        default configuration (notably its congestion-controlled bitrate),
        which does not otherwise touch these two properties.
        """
        factory = encoder.get_factory()
        factory_name = factory.get_name() if factory else ""
        if factory_name == "opusenc":
            if encoder.find_property("inband-fec") is not None:
                encoder.set_property("inband-fec", True)
            if encoder.find_property("packet-loss-percentage") is not None:
                encoder.set_property(
                    "packet-loss-percentage", self.TX_OPUS_FEC_LOSS_PERC
                )
            self._logger.info(
                f"opusenc tuned for {consumer_id}: inband-fec=True, "
                f"packet-loss-percentage={self.TX_OPUS_FEC_LOSS_PERC}"
            )
        return False

    def _consumer_added(
        self,
        webrtcsink: Gst.Bin,
        peer_id: str,
        webrtcbin: Gst.Element,
    ) -> None:
        self._logger.info(f"consumer added with peer id: {peer_id}")

        # Gst.debug_bin_to_dot_file(
        #     self._pipeline_sender, Gst.DebugGraphDetails.ALL, "pipeline_full"
        # )

        GLib.timeout_add_seconds(5, self._dump_latency)

        self._setup_data_channel(peer_id, webrtcbin)

        # Deepen this consumer's receive jitter buffer before media flows
        # so transient Wi-Fi jitter on the phone->robot voice leg doesn't
        # starve the speaker. Must be set on `webrtcbin` here, while it is
        # still being negotiated, for the internal jitterbuffer to pick it
        # up. See `RX_JITTER_LATENCY_MS`.
        webrtcbin.set_property("latency", self.RX_JITTER_LATENCY_MS)

        # Make audio bidirectional before SDP offer is generated
        self._enable_audio_receive(webrtcbin)

        # Listen for incoming audio pads from the browser (bidirectional audio)
        webrtcbin.connect("pad-added", self._on_consumer_pad_added, peer_id)

        # Watchdog wiring: track ICE / connection / signaling state on
        # this peer's webrtcbin so we can detect a stuck negotiation
        # and report it (instead of letting the JS client spin
        # forever). See `ICE_NEGOTIATION_DEADLINE_S`.
        self._install_negotiation_watchdog(peer_id, webrtcbin)

    # GstWebRTCRTPTransceiverDirection enum values
    _WEBRTC_DIRECTION_SENDRECV = 4

    def _enable_audio_receive(self, webrtcbin: Gst.Element) -> None:
        """Set all transceivers to sendrecv for bidirectional audio.

        Must be called before the SDP offer is generated (in consumer-added).

        All transceivers are set to sendrecv.  For the video transceiver
        this is harmless when the peer is a browser (it answers recvonly).
        When the peer is ``webrtcsrc`` (Python SDK), the video sendrecv
        causes it to create an internal appsrc that emits a non-fatal
        ``not-negotiated`` error — the client's bus-message handler
        ignores this specific error so the pipeline stays alive.
        """
        i = 0
        while True:
            trans = webrtcbin.emit("get-transceiver", i)
            if trans is None:
                break

            current_dir = trans.get_property("direction")
            trans.set_property("direction", self._WEBRTC_DIRECTION_SENDRECV)
            new_dir = trans.get_property("direction")
            self._logger.info(f"Transceiver {i}: {current_dir} -> {new_dir}")
            i += 1

    def _consumer_removed(
        self,
        webrtcsink: Gst.Bin,
        peer_id: str,
        webrtcbin: Gst.Element,
    ) -> None:
        self._logger.info(f"consumer removed: {peer_id}")
        self._cleanup_incoming_audio(peer_id)
        # Cancel any outstanding watchdog for this peer; the consumer
        # is gone so there's nothing left to police.
        self._teardown_negotiation_watchdog(peer_id)
        if self._on_peer_disconnect is not None:
            try:
                self._on_peer_disconnect(peer_id)
            except Exception as e:
                self._logger.warning(
                    f"peer-disconnect handler raised for {peer_id}: {e}"
                )

    def _on_consumer_pad_added(
        self,
        webrtcbin: Gst.Element,
        pad: Gst.Pad,
        peer_id: str,
    ) -> None:
        """Handle incoming pads from the browser for bidirectional audio.

        We cannot add elements to _pipeline_sender because webrtcsink manages
        that pipeline internally and dynamic additions crash the connection.
        Instead we use a pad probe to intercept RTP buffers and forward them
        to a completely separate playback pipeline via appsrc.
        """
        if pad.get_direction() != Gst.PadDirection.SRC:
            return

        pad_name = pad.get_name()
        caps = pad.get_current_caps()
        if caps is None:
            caps = pad.query_caps(None)

        self._logger.info(
            f"Consumer pad: {pad_name}, caps: {caps.to_string() if caps else 'none'}"
        )

        if caps is None or caps.get_size() == 0:
            return

        struct = caps.get_structure(0)
        media = struct.get_string("media") if struct.has_field("media") else ""
        if media != "audio":
            return

        self._logger.info(f"Setting up incoming audio playback for peer {peer_id}")

        # Build playback pipeline element-by-element
        self._pipeline_playback = Gst.Pipeline.new(f"audio_playback_{peer_id}")

        sender_clock = self._pipeline_sender.get_pipeline_clock()
        self._pipeline_playback.use_clock(sender_clock)
        self._pipeline_playback.set_start_time(Gst.CLOCK_TIME_NONE)

        appsrc = Gst.ElementFactory.make("appsrc", self.INCOMING_AUDIO_SRC_NAME)
        appsrc.set_property("format", Gst.Format.TIME)
        appsrc.set_property("is-live", True)
        appsrc.set_property("caps", caps)

        rtpopusdepay = Gst.ElementFactory.make("rtpopusdepay")
        opusdec = Gst.ElementFactory.make("opusdec")
        # Wi-Fi resilience on the phone->robot voice leg. The browser
        # encoder emits Opus in-band FEC (a redundant copy of the
        # previous frame piggybacked on the next packet) and ramps it
        # with the loss it sees over RTCP; without these two properties
        # the decoder silently ignores that redundancy and we glitch on
        # every dropped packet. `use-inband-fec` reconstructs the lost
        # frame from the next one (one packet of look-ahead, so the
        # latency cost is ~one frame); `plc` conceals whatever FEC can't
        # recover. This is the cheapest robustness win on this path.
        opusdec.set_property("use-inband-fec", True)
        opusdec.set_property("plc", True)

        audiosink = self._build_audiosink_element()
        if audiosink is None:
            self._logger.error("Failed to create audio sink element")
            return
        audiosink.set_property("sync", True)

        # Per-branch audioconvert+audioresample so the wobbler appsink's
        # F32LE/2/16000 caps don't drag the audiosink branch into a rate
        # the device can't accept (e.g. wireless XMOS PCM falls back to
        # IEC958 at non-native rates).
        tee = Gst.ElementFactory.make("tee")
        queue_speaker = Gst.ElementFactory.make("queue")
        ac_speaker = Gst.ElementFactory.make("audioconvert")
        ar_speaker = Gst.ElementFactory.make("audioresample")
        queue_wobbler = Gst.ElementFactory.make("queue")
        ac_wobbler = Gst.ElementFactory.make("audioconvert")
        ar_wobbler = Gst.ElementFactory.make("audioresample")

        appsink_wobbler = self._make_wobbler_appsink()

        # Software AEC far-end reference: tap the speaker branch (what is
        # physically played) with the shared echo probe. opusdec emits S16LE@48k,
        # which the probe accepts, so no convert/resample is needed. The probe
        # is reused across peers, so detach it from any previous pipeline first.
        # Caveat: there is one probe (webrtcdsp binds a single far-end), so with
        # several peers streaming at once AEC only references the most recently
        # connected one. Fine for the usual single-conversation case.
        webrtcechoprobe = self._webrtcechoprobe if self._aec_enabled else None
        if webrtcechoprobe is not None:
            old_parent = webrtcechoprobe.get_parent()
            if old_parent is not None:
                old_parent.remove(webrtcechoprobe)

        for elem in [
            appsrc,
            rtpopusdepay,
            opusdec,
            tee,
            queue_speaker,
            *([webrtcechoprobe] if webrtcechoprobe is not None else []),
            ac_speaker,
            ar_speaker,
            audiosink,
            queue_wobbler,
            ac_wobbler,
            ar_wobbler,
            appsink_wobbler,
        ]:
            self._pipeline_playback.add(elem)
        appsrc.link(rtpopusdepay)
        rtpopusdepay.link(opusdec)
        opusdec.link(tee)
        tee.link(queue_speaker)
        if webrtcechoprobe is not None:
            queue_speaker.link(webrtcechoprobe)
            webrtcechoprobe.link(ac_speaker)
        else:
            queue_speaker.link(ac_speaker)
        ac_speaker.link(ar_speaker)
        ar_speaker.link(audiosink)
        tee.link(queue_wobbler)
        queue_wobbler.link(ac_wobbler)
        ac_wobbler.link(ar_wobbler)
        ar_wobbler.link(appsink_wobbler)

        play_bus = self._pipeline_playback.get_bus()
        play_bus.add_watch(
            GLib.PRIORITY_DEFAULT, self._on_bus_message, self._pipeline_playback
        )

        self._pipeline_playback.set_state(Gst.State.PAUSED)
        self._pipeline_playback.set_base_time(self._pipeline_sender.get_base_time())
        self._pipeline_playback.set_state(Gst.State.PLAYING)

        # Pad probe: intercept every RTP buffer, forward to the separate
        # playback pipeline, then DROP so webrtcsink's pipeline is unaffected.
        def _buffer_probe(pad: Gst.Pad, info: Gst.PadProbeInfo, _: None) -> int:
            buf = info.get_buffer()
            appsrc.push_buffer(buf)
            return int(Gst.PadProbeReturn.DROP)

        probe_id = pad.add_probe(Gst.PadProbeType.BUFFER, _buffer_probe, None)

        if self._head_wobbler is not None:
            self._head_wobbler.reset()
            self._head_wobbler.start()

        self._incoming_audio[peer_id] = {
            "playback_pipeline": self._pipeline_playback,
            "probe_id": probe_id,
            "pad": pad,
        }
        self._logger.info(f"Audio playback pipeline started for peer {peer_id}")

    def _on_playback_bus_message(
        self, bus: Gst.Bus, msg: Gst.Message, peer_id: str
    ) -> bool:
        """Handle messages from a per-peer audio playback pipeline."""
        if msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            self._logger.error(f"Audio playback error for {peer_id}: {err} {debug}")
            return False
        if msg.type == Gst.MessageType.EOS:
            self._logger.info(f"Audio playback EOS for {peer_id}")
            return False
        return True

    def _cleanup_incoming_audio(self, peer_id: str) -> None:
        """Remove the incoming-audio pad probe and playback pipeline for a peer."""
        info = self._incoming_audio.pop(peer_id, None)
        if info is None:
            return

        pad = info.get("pad")
        probe_id = info.get("probe_id")
        if pad is not None and probe_id is not None:
            pad.remove_probe(probe_id)

        playback_pipe = info.get("playback_pipeline")
        if playback_pipe is not None:
            playback_pipe.set_state(Gst.State.NULL)
            # Rescue the shared echo probe so it survives for the next peer.
            if (
                self._webrtcechoprobe is not None
                and self._webrtcechoprobe.get_parent() is playback_pipe
            ):
                playback_pipe.remove(self._webrtcechoprobe)
        self._logger.info(f"Cleaned up incoming audio for peer {peer_id}")

    def clear_incoming_audio(self) -> None:
        """Flush queued/rendering audio in the incoming-audio playback pipeline.

        Used for barge-in: drops audio already received from a WebRTC client
        and queued for the robot's speaker so the robot stops speaking promptly.

        The playback pipeline shares the sender clock + base-time, so incoming
        buffer PTS live in that shared running-time; we flush with
        ``reset_time=False`` to keep the timeline intact (``reset_time=True``
        would strand future-stamped buffers and stall playback). The pad probe
        keeps pushing new RTP buffers into the appsrc, which resume in sync.
        """
        pipeline = self._pipeline_playback
        if pipeline is None:
            self._logger.info("No incoming-audio pipeline to clear.")
            return
        appsrc = pipeline.get_by_name(self.INCOMING_AUDIO_SRC_NAME)
        if appsrc is None:
            self._logger.warning("Incoming-audio appsrc not found; nothing to flush.")
            return
        appsrc.send_event(Gst.Event.new_flush_start())
        appsrc.send_event(Gst.Event.new_flush_stop(reset_time=False))
        if self._head_wobbler is not None:
            self._head_wobbler.reset()
        self._logger.info("Flushed incoming audio playback")

    @property
    def resolution(self) -> tuple[int, int]:
        """Get the current camera resolution as a tuple (width, height)."""
        return (self._resolution.value[0], self._resolution.value[1])

    @property
    def framerate(self) -> int:
        """Get the current camera framerate."""
        return self._resolution.value[2]

    def _configure_video(
        self, cam_path: str, pipeline: Gst.Pipeline, webrtcsink: Gst.Element
    ) -> None:
        """Configure the video pipeline based on the detected camera and platform.

        The pipeline is structured as:

            camera_source → [optional decode/convert] → tee
                ├─ IPC branch  (unixfdsink on Linux/macOS, win32ipcvideosink on Windows)
                └─ WebRTC branch:
                     RPi (imx708): v4l2h264enc → capsfilter_h264 → webrtcsink
                     Others:       raw video → webrtcsink (encodes internally)

        Args:
            cam_path: Camera device path or special value ('' for no camera,
                'use_sim' for MuJoCo UDP, 'imx708' for RPi CSI, or a platform
                device path).
            pipeline: The GStreamer pipeline to add elements to.
            webrtcsink: The webrtcsink element for WebRTC output.

        """
        self._logger.debug(f"Configuring video: cam_path={cam_path}")

        # --- No camera ---
        if cam_path == "":
            self._logger.warning("No camera detected. Video pipeline not configured.")
            return

        # --- Build the camera source chain (ends with raw video) ---
        source_elements: list[Gst.Element] = []

        if cam_path == "use_sim":
            # TODO: Optimize later by feeding frames directly via appsrc
            # instead of UDP loopback.
            source_elements = self._build_sim_source()

        elif cam_path == "use_mockup_sim":
            source_elements = self._build_autovideo_source()

        elif cam_path == "imx708":
            # Raspberry Pi CSI camera (libcamerasrc)
            source_elements = self._build_libcamera_source()

        elif platform.system() == "Windows":
            source_elements = self._build_windows_source(cam_path)

        elif platform.system() == "Darwin":
            source_elements = self._build_macos_source(cam_path)

        else:
            # Linux V4L2 USB camera
            source_elements = self._build_v4l2_source(cam_path)

        if not source_elements:
            self._logger.error("Failed to create camera source elements.")
            return

        # --- Add all source elements to the pipeline and link them ---
        for elem in source_elements:
            pipeline.add(elem)
        for i in range(len(source_elements) - 1):
            source_elements[i].link(source_elements[i + 1])

        last_source = source_elements[-1]

        is_rpi = cam_path == "imx708"

        # Pin the raw video format before the tee so that both branches
        # receive a known format.  On non-RPi paths this must be a format
        # that differs from BGR (the IPC branch output format), otherwise
        # the IPC-branch videoconvert would run in passthrough mode and
        # skip the FD-backed buffer re-allocation that unixfdsink requires.
        if not is_rpi:
            caps_raw = Gst.Caps.from_string(
                f"video/x-raw,format=I420,"
                f"width={self.resolution[0]},"
                f"height={self.resolution[1]},"
                f"framerate={self.framerate}/1"
            )
            capsfilter_raw = Gst.ElementFactory.make("capsfilter", "pre_tee_caps")
            capsfilter_raw.set_property("caps", caps_raw)
            pipeline.add(capsfilter_raw)
            last_source.link(capsfilter_raw)
            last_source = capsfilter_raw

        # --- Tee: split into IPC + WebRTC branches ---
        tee = Gst.ElementFactory.make("tee")
        pipeline.add(tee)
        last_source.link(tee)

        # IPC branch: share camera with local applications
        self._build_ipc_branch(tee, pipeline, is_rpi=is_rpi)

        # WebRTC branch
        queue_webrtc = Gst.ElementFactory.make("queue", "queue_webrtc")
        pipeline.add(queue_webrtc)
        tee.link(queue_webrtc)

        if is_rpi:
            # RPi: use hardware H264 encoder (webrtcsink doesn't have v4l2h264enc)
            self._build_rpi_encoder_branch(queue_webrtc, pipeline, webrtcsink)
        else:
            # All other platforms: feed raw video, let webrtcsink handle encoding
            queue_webrtc.link(webrtcsink)

    def _build_sim_source(self) -> list[Gst.Element]:
        """Build source chain for MuJoCo simulation (UDP raw video)."""
        udpsrc = Gst.ElementFactory.make("udpsrc")
        udpsrc.set_property("port", 5005)
        caps = Gst.Caps.from_string(
            "application/x-rtp,media=(string)video,clock-rate=(int)90000,"
            "encoding-name=(string)RAW,sampling=(string)RGB,depth=(string)8,"
            f"width=(string){self.resolution[0]},height=(string){self.resolution[1]},"
            "payload=(int)96"
        )
        udpsrc.set_property("caps", caps)

        queue = Gst.ElementFactory.make("queue")
        rtpvrawdepay = Gst.ElementFactory.make("rtpvrawdepay")
        videoconvert = Gst.ElementFactory.make("videoconvert")
        videorate = Gst.ElementFactory.make("videorate")

        elements = [udpsrc, queue, rtpvrawdepay, videoconvert, videorate]
        if not all(elements):
            raise RuntimeError("Failed to create simulation video source elements")
        return elements

    def _build_autovideo_source(self) -> list[Gst.Element]:
        """Build source chain using autovideosrc (auto-detect system camera).

        Used by mockup simulation to grab video from whatever camera is
        available on the host system.
        """
        camsrc = Gst.ElementFactory.make("autovideosrc")
        videoconvert = Gst.ElementFactory.make("videoconvert")
        videorate = Gst.ElementFactory.make("videorate")

        caps_raw = Gst.Caps.from_string(
            f"video/x-raw,width={self.resolution[0]},"
            f"height={self.resolution[1]},"
            f"framerate={self.framerate}/1"
        )
        capsfilter = Gst.ElementFactory.make("capsfilter")
        capsfilter.set_property("caps", caps_raw)

        elements = [camsrc, videoconvert, videorate, capsfilter]
        if not all(elements):
            raise RuntimeError("Failed to create autovideo source elements")
        return elements

    def _build_libcamera_source(self) -> list[Gst.Element]:
        """Build source chain for RPi CSI camera (libcamerasrc)."""
        camerasrc = Gst.ElementFactory.make("libcamerasrc")
        caps = Gst.Caps.from_string(
            f"video/x-raw,width={self.resolution[0]},height={self.resolution[1]},"
            f"framerate={self.framerate}/1,format=YUY2,"
            "colorimetry=bt709,interlace-mode=progressive"
        )
        capsfilter = Gst.ElementFactory.make("capsfilter")
        capsfilter.set_property("caps", caps)

        elements = [camerasrc, capsfilter]
        if not all(elements):
            raise RuntimeError("Failed to create libcamerasrc elements")
        return elements

    def _build_v4l2_source(self, device_path: str) -> list[Gst.Element]:
        """Build source chain for Linux V4L2 USB camera.

        A capsfilter is placed after v4l2src to explicitly negotiate the
        MJPEG format, resolution and framerate.  Without it, v4l2src may
        auto-negotiate an undesirable format (e.g. YUYV at low fps).
        """
        camsrc = Gst.ElementFactory.make("v4l2src")
        camsrc.set_property("device", device_path)

        caps_mjpeg = Gst.Caps.from_string(
            f"image/jpeg,width={self.resolution[0]},"
            f"height={self.resolution[1]},"
            f"framerate={self.framerate}/1"
        )
        capsfilter = Gst.ElementFactory.make("capsfilter")
        capsfilter.set_property("caps", caps_mjpeg)

        queue = Gst.ElementFactory.make("queue")
        jpegdec = Gst.ElementFactory.make("jpegdec")
        videoconvert = Gst.ElementFactory.make("videoconvert")

        elements = [camsrc, capsfilter, queue, jpegdec, videoconvert]
        if not all(elements):
            raise RuntimeError("Failed to create V4L2 video source elements")
        return elements

    def _build_windows_source(self, device_name: str) -> list[Gst.Element]:
        """Build source chain for Windows Media Foundation camera.

        A capsfilter is placed after mfvideosrc to explicitly negotiate the
        MJPEG format, resolution and framerate.  Without it, mfvideosrc may
        auto-negotiate a format (e.g. NV12 at a low fps) that cannot satisfy
        the downstream I420 capsfilter, causing the source to error out with
        "streaming stopped, reason error (-5)".
        """
        camsrc = Gst.ElementFactory.make("mfvideosrc")
        camsrc.set_property("device-name", device_name)

        caps_mjpeg = Gst.Caps.from_string(
            f"image/jpeg,width={self.resolution[0]},"
            f"height={self.resolution[1]},"
            f"framerate={self.framerate}/1"
        )
        capsfilter = Gst.ElementFactory.make("capsfilter")
        capsfilter.set_property("caps", caps_mjpeg)

        queue = Gst.ElementFactory.make("queue")
        jpegdec = Gst.ElementFactory.make("jpegdec")
        videoconvert = Gst.ElementFactory.make("videoconvert")

        elements = [camsrc, capsfilter, queue, jpegdec, videoconvert]
        if not all(elements):
            raise RuntimeError("Failed to create Windows video source elements")
        return elements

    def _build_macos_source(self, device_index: str) -> list[Gst.Element]:
        """Build source chain for macOS AVFoundation camera.

        Unlike v4l2src and mfvideosrc which expose raw MJPEG streams,
        avfvideosrc typically decodes MJPEG internally and presents raw
        video to GStreamer.  A capsfilter pins the resolution and
        framerate so that auto-negotiation does not pick an undesirable
        mode (e.g. low-fps NV12 instead of the camera's preferred
        1920x1080@60).
        """
        camsrc = Gst.ElementFactory.make("avfvideosrc")
        camsrc.set_property("device-index", int(device_index))

        caps_raw = Gst.Caps.from_string(
            f"video/x-raw,width={self.resolution[0]},"
            f"height={self.resolution[1]},"
            f"framerate={self.framerate}/1"
        )
        capsfilter = Gst.ElementFactory.make("capsfilter")
        capsfilter.set_property("caps", caps_raw)

        queue = Gst.ElementFactory.make("queue")
        videoconvert = Gst.ElementFactory.make("videoconvert")

        elements = [camsrc, capsfilter, queue, videoconvert]
        if not all(elements):
            raise RuntimeError("Failed to create macOS video source elements")
        return elements

    def _build_ipc_branch(
        self, tee: Gst.Element, pipeline: Gst.Pipeline, *, is_rpi: bool = False
    ) -> None:
        """Build the IPC branch for sharing camera with local applications.

        Linux/macOS: unixfdsink at CAMERA_SOCKET_PATH
        Windows: win32ipcvideosink with CAMERA_PIPE_NAME

        On RPi (libcamerasrc) the camera produces dmabuf-backed buffers, so
        ``unixfdsink`` works directly behind the ``tee``.

        On all other platforms the camera chain produces regular system-memory
        buffers.  ``unixfdsink`` requires FD-backed (memfd) buffers and
        proposes a ``GstShmAllocator`` via the allocation query.  However,
        when it sits behind a ``tee`` together with ``webrtcsink``, the
        allocation query is won by ``webrtcsink`` and upstream elements
        allocate system-memory buffers instead — causing ``unixfdsink`` to
        reject them with *"Expecting buffers with FD memories"*.

        The fix inserts two elements before ``unixfdsink``:

        1. ``identity drop-allocation=true`` — blocks the conflicting
           allocation query from propagating upstream through the ``tee``,
           so ``unixfdsink`` can negotiate its own allocator independently.
        2. ``videoconvert`` with a forced output format (BGR) — because the
           allocation query is blocked, ``videoconvert`` honours
           ``unixfdsink``'s ``GstShmAllocator`` proposal and allocates new
           output buffers backed by ``memfd``.  A format change (e.g.
           I420 → BGR) is required to prevent ``videoconvert`` from running
           in passthrough mode (which would just forward the original
           system-memory buffer unchanged).

        Using BGR as the IPC format has the added benefit that the client-
        side reader (``camera_gstreamer.py``) can consume frames directly
        without an extra ``videoconvert`` step.
        """
        queue_ipc = Gst.ElementFactory.make("queue", "queue_ipc")
        pipeline.add(queue_ipc)
        tee.link(queue_ipc)

        if platform.system() == "Windows":
            ipc_sink = Gst.ElementFactory.make("win32ipcvideosink")
            if ipc_sink is None:
                self._logger.warning(
                    "win32ipcvideosink not available. "
                    "Local camera IPC will not work on Windows."
                )
                return
            ipc_sink.set_property("pipe-name", CAMERA_PIPE_NAME)
        else:
            # Linux and macOS both use unixfdsink
            ipc_sink = Gst.ElementFactory.make("unixfdsink")
            if ipc_sink is None:
                self._logger.warning(
                    "unixfdsink not available. Local camera IPC will not work."
                )
                return
            if is_local_camera_available():
                # Prevent crash if socket already exists
                os.remove(CAMERA_SOCKET_PATH)
            ipc_sink.set_property("socket-path", CAMERA_SOCKET_PATH)

        # On RPi, libcamerasrc produces dmabuf FD-backed buffers natively,
        # so unixfdsink works directly.  On other platforms we need the
        # identity + videoconvert workaround described above.
        if is_rpi:
            pipeline.add(ipc_sink)
            queue_ipc.link(ipc_sink)
        else:
            identity = Gst.ElementFactory.make("identity", "ipc_identity")
            identity.set_property("drop-allocation", True)

            videoconvert_ipc = Gst.ElementFactory.make(
                "videoconvert", "ipc_videoconvert"
            )

            caps_bgr = Gst.Caps.from_string(
                f"video/x-raw,format=BGR,"
                f"width={self.resolution[0]},"
                f"height={self.resolution[1]},"
                f"framerate={self.framerate}/1"
            )
            capsfilter_ipc = Gst.ElementFactory.make("capsfilter", "ipc_capsfilter")
            capsfilter_ipc.set_property("caps", caps_bgr)

            for elem in [identity, videoconvert_ipc, capsfilter_ipc, ipc_sink]:
                pipeline.add(elem)

            queue_ipc.link(identity)
            identity.link(videoconvert_ipc)
            videoconvert_ipc.link(capsfilter_ipc)
            capsfilter_ipc.link(ipc_sink)

    def _build_rpi_encoder_branch(
        self,
        queue_webrtc: Gst.Element,
        pipeline: Gst.Pipeline,
        webrtcsink: Gst.Element,
    ) -> None:
        """Build the RPi hardware H264 encoder branch.

        webrtcsink does not have v4l2h264enc, so we encode explicitly on RPi.
        """
        v4l2h264enc = Gst.ElementFactory.make("v4l2h264enc")
        extra_controls_structure = Gst.Structure.new_empty("extra-controls")
        extra_controls_structure.set_value("repeat_sequence_header", 1)
        extra_controls_structure.set_value("video_bitrate", 5_000_000)
        extra_controls_structure.set_value("h264_i_frame_period", 60)
        extra_controls_structure.set_value("video_gop_size", 256)
        v4l2h264enc.set_property("extra-controls", extra_controls_structure)

        # H264 Level 3.1 + Constrained Baseline for Safari/WebKit compatibility
        caps_h264 = Gst.Caps.from_string(
            "video/x-h264,stream-format=byte-stream,alignment=au,"
            "level=(string)3.1,profile=(string)constrained-baseline"
        )
        capsfilter_h264 = Gst.ElementFactory.make("capsfilter")
        capsfilter_h264.set_property("caps", caps_h264)

        if not all([v4l2h264enc, capsfilter_h264]):
            raise RuntimeError("Failed to create RPi H264 encoder elements")

        pipeline.add(v4l2h264enc)
        pipeline.add(capsfilter_h264)

        queue_webrtc.link(v4l2h264enc)
        v4l2h264enc.link(capsfilter_h264)
        capsfilter_h264.link(webrtcsink)

    def _configure_audio(self, pipeline: Gst.Pipeline, webrtcsink: Gst.Element) -> None:
        """Configure the audio capture pipeline.

        Detects the audio device based on the platform and feeds it into
        webrtcsink for WebRTC streaming.  When no audio source is available
        (e.g. missing Reachy Mini Audio USB card on the wireless CM4), the
        audio branch is skipped entirely so that the failure does not tear
        down the shared webrtcsink pipeline — and with it, the video branch.
        """
        self._logger.debug("Configuring audio")

        audiosrc = self._build_audio_source()
        if audiosrc is None:
            self._logger.warning(
                "No audio source available. "
                "Streaming video only; audio will be unavailable."
            )
            return

        queue = Gst.ElementFactory.make("queue", "queue_audiosrc")
        pipeline.add(audiosrc)
        pipeline.add(queue)

        # Software AEC on the autoaudiosrc fallback (no Reachy Mini card → no
        # XMOS hardware AEC). webrtcdsp subtracts the far-end reference captured
        # by the paired webrtcechoprobe from the mic signal. The probe must
        # exist before webrtcdsp starts, so create it here; both elements are
        # needed, so disable AEC if either is unavailable.
        self._aec_enabled = False
        webrtcdsp = None
        factory = audiosrc.get_factory()
        factory_name = factory.get_name() if factory else ""
        if factory_name == "autoaudiosrc":
            if self._webrtcechoprobe is None:
                self._webrtcechoprobe = Gst.ElementFactory.make("webrtcechoprobe")
                if self._webrtcechoprobe is not None:
                    self._webrtcechoprobe.set_property("name", AEC_PROBE_NAME)
            if self._webrtcechoprobe is not None:
                webrtcdsp = Gst.ElementFactory.make("webrtcdsp")
            if webrtcdsp is None:
                self._logger.warning(
                    "webrtcdsp/webrtcechoprobe unavailable; "
                    "software echo cancellation disabled."
                )

        if webrtcdsp is not None:
            self._aec_enabled = True
            webrtcdsp.set_property("probe", AEC_PROBE_NAME)
            # webrtcdsp requires S16LE at 8/16/32/48 kHz.
            chain = self._make_aec_caps_chain()
            for el in (*chain, webrtcdsp):
                pipeline.add(el)
            audiosrc.link(chain[0])
            for upstream, downstream in zip(chain, chain[1:]):
                upstream.link(downstream)
            chain[-1].link(webrtcdsp)
            webrtcdsp.link(queue)
            self._logger.info("No hardware AEC; enabled software echo cancellation.")
        else:
            audiosrc.link(queue)

        # Link into webrtcsink last, once the full upstream chain exists, so its
        # request pad / stream discovery sees a fully-linked input.
        queue.link(webrtcsink)

    def _make_aec_caps_chain(self) -> list[Gst.Element]:
        """Build the convert/resample/caps chain feeding an AEC element.

        ``webrtcdsp`` and ``webrtcechoprobe`` only accept S16LE at
        8/16/32/48 kHz, so each is fronted by
        ``audioconvert → audioresample → capsfilter(S16LE @ AEC_RATE)``.

        Returns:
            The three elements in link order; the caller adds them to its
            pipeline and links the last one into the AEC element.

        """
        audioconvert = Gst.ElementFactory.make("audioconvert")
        audioresample = Gst.ElementFactory.make("audioresample")
        capsfilter = Gst.ElementFactory.make("capsfilter")
        capsfilter.set_property(
            "caps",
            Gst.Caps.from_string(
                f"audio/x-raw,format=S16LE,rate={AEC_RATE},"
                f"channels={AEC_CHANNELS},layout=interleaved"
            ),
        )
        return [audioconvert, audioresample, capsfilter]

    def _selected_device_name(self, direction: str) -> Optional[str]:
        """Return the selected device name for direction (input/output), or None."""
        from reachy_mini.daemon.app.routers.audio_devices import (
            get_local_selected_input,
            get_local_selected_output,
        )

        # Read the in-process selection directly (no self-HTTP call): the media
        # server runs inside the daemon.
        return (
            get_local_selected_input()
            if direction == "input"
            else get_local_selected_output()
        )

    def _build_audio_source(self) -> Optional[Gst.Element]:
        """Build a platform-aware audio source element.

        Detection order:
        1. .asoundrc — on the wireless CM4 the ``.asoundrc`` file defines
           ALSA aliases (``reachymini_audio_src``).  When present we use
           ``alsasrc`` directly, matching the behaviour of the original
           daemon and avoiding PipeWire clock issues.  The Reachy Mini
           Audio USB card is checked first: if it is absent, the ALSA
           aliases in ``.asoundrc`` resolve to a missing ``hw:`` card and
           ``alsasrc`` would fail to open at pipeline start, tearing
           down the shared webrtcsink pipeline (video included).
        2. GstDeviceMonitor — finds the Reachy Mini Audio card by name and
           returns a platform-native element (pulsesrc, wasapi2src,
           osxaudiosrc).  Used on Lite and desktop platforms.
        3. autoaudiosrc — last resort when no Reachy Mini card is found.

        Returns:
            A GStreamer audio source element, or None if no audio is available.

        """
        # An explicit user selection (via the audio-devices API) overrides the
        # .asoundrc / auto-detected source. It is resolved platform-correctly
        # below via get_audio_device's target_name.
        selected = self._selected_device_name("input")

        # Wireless CM4 (no explicit selection): .asoundrc defines the
        # reachymini_audio_src alias routing through the XMOS AEC loopback.
        if selected is None and has_reachymini_asoundrc():
            respeaker = init_respeaker_usb()
            if respeaker is None:
                self._logger.warning(
                    "Reachy Mini Audio USB device not found — "
                    "skipping audio capture to keep the video pipeline alive."
                )
                return None
            respeaker.close()

            audiosrc = Gst.ElementFactory.make("alsasrc")
            audiosrc.set_property("device", "reachymini_audio_src")
            self._logger.info("Using ALSA device reachymini_audio_src for capture.")
            return audiosrc

        id_audio_card = get_audio_device(
            "Source", target_name=selected or DEFAULT_AUDIO_TARGET
        )

        if id_audio_card is not None:
            if platform.system() == "Windows":
                audiosrc = Gst.ElementFactory.make("wasapi2src")
                audiosrc.set_property("device", id_audio_card)
                self._logger.info(f"Using WASAPI device {id_audio_card} for capture.")
            elif platform.system() == "Darwin":
                audiosrc = Gst.ElementFactory.make("osxaudiosrc")
                audiosrc.set_property("unique-id", id_audio_card)
                self._logger.info(
                    f"Using CoreAudio device {id_audio_card} for capture."
                )
            else:
                audiosrc = Gst.ElementFactory.make("pulsesrc")
                audiosrc.set_property("device", f"{id_audio_card}")
                self._logger.info(
                    f"Using PulseAudio/PipeWire device {id_audio_card} for capture."
                )
            return audiosrc

        self._logger.warning(
            "No Reachy Mini audio card found, using default audio source."
        )
        return Gst.ElementFactory.make("autoaudiosrc")

    def _on_bus_message(
        self, bus: Gst.Bus, msg: Gst.Message, pipeline: Gst.Pipeline
    ) -> bool:
        return handle_default_bus_message(self._logger, msg, pipeline)

    def start(self) -> None:
        """Rebuild the pipeline from scratch and start it.

        Rebuilding ensures a clean state after stop() released all hardware.
        """
        self._logger.debug("Starting WebRTC (rebuilding pipeline)")
        self._build_pipeline()
        self._pipeline_sender.set_state(Gst.State.PLAYING)
        GLib.timeout_add_seconds(5, self._dump_latency)

    def stop(self) -> None:
        """Stop the pipeline and release all hardware (camera, audio)."""
        self._logger.debug("Stopping WebRTC")
        self._pipeline_sender.set_state(Gst.State.NULL)

    def play_sound(self, sound_file: str) -> None:
        """Play a sound file on the robot's speaker.

        Uses GStreamer's playbin element with a platform-aware audio sink.
        This is used for daemon-side sounds (wake-up, sleep, etc.).

        Args:
            sound_file: Path to the sound file to play. If the file is not
                found at the given path, it is looked up in the assets directory.

        """
        if not os.path.exists(sound_file):
            file_path = f"{ASSETS_ROOT_PATH}/{sound_file}"
            if not os.path.exists(file_path):
                self._logger.error(
                    f"Sound file {sound_file} not found in assets directory "
                    "or given path."
                )
                return
        else:
            file_path = sound_file

        if self._playbin is not None:
            self._playbin.set_state(Gst.State.NULL)

        playbin = Gst.ElementFactory.make("playbin", "player")
        if not playbin:
            self._logger.error("Failed to create playbin element")
            return

        # Build file URI
        if os.name == "nt":
            uri_path = file_path.replace("\\", "/")
            if not uri_path.startswith("/") and ":" in uri_path:
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

    def stop_sound(self) -> None:
        """Stop the currently playing sound file.

        If no sound is currently playing this is a no-op.
        """
        if self._playbin is not None:
            self._playbin.set_state(Gst.State.NULL)
            self._playbin = None

    def _build_audiosink_element(self) -> Optional[Gst.Element]:
        """Build a platform-aware audio sink GStreamer element.

        Same detection order as ``_build_audio_source()``: .asoundrc first
        (wireless CM4), then DeviceMonitor, then autoaudiosink.

        Returns:
            A GStreamer audio sink element, or None to use the default.

        """
        # An explicit user selection (via the audio-devices API) overrides the
        # .asoundrc / auto-detected sink. It is resolved platform-correctly
        # below via get_audio_device's target_name.
        selected = self._selected_device_name("output")

        # Wireless CM4 (no explicit selection): .asoundrc defines the
        # reachymini_audio_sink alias.
        if selected is None and has_reachymini_asoundrc():
            audiosink = Gst.ElementFactory.make("alsasink")
            audiosink.set_property("device", "reachymini_audio_sink")
            self._logger.info("Using ALSA device reachymini_audio_sink for playback.")
            return audiosink

        id_audio_card = get_audio_device(
            "Sink", target_name=selected or DEFAULT_AUDIO_TARGET
        )

        if id_audio_card is not None:
            if platform.system() == "Windows":
                audiosink = Gst.ElementFactory.make("wasapi2sink")
                audiosink.set_property("device", id_audio_card)
                self._logger.info(f"Using WASAPI device {id_audio_card} for playback.")
            elif platform.system() == "Darwin":
                audiosink = Gst.ElementFactory.make("osxaudiosink")
                audiosink.set_property("unique-id", id_audio_card)
                self._logger.info(
                    f"Using CoreAudio device {id_audio_card} for playback."
                )
            else:
                audiosink = Gst.ElementFactory.make("pulsesink")
                audiosink.set_property("device", f"{id_audio_card}")
                self._logger.info(
                    f"Using PulseAudio/PipeWire device {id_audio_card} for playback."
                )
            return audiosink

        return Gst.ElementFactory.make("autoaudiosink")

    def _make_wobbler_appsink(self) -> Gst.Element:
        """Create a sync=True appsink that feeds audio to the head wobbler.

        new-sample fires at the buffer's PTS on the pipeline clock —
        the same instant the audiosink renders that audio.
        """
        appsink = Gst.ElementFactory.make("appsink")
        # Force mono so the speech tapper receives a 1-D float32 array.
        # The per-branch audioconvert handles the downmix.
        caps = Gst.Caps.from_string(
            f"audio/x-raw,format=F32LE,channels=1,rate={self.WOBBLER_SAMPLE_RATE},layout=interleaved"
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

        The appsink is sync=True so the callback fires at the buffer's
        PTS on the pipeline clock — audio is playing NOW.
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
        """Build a Gst.Bin splitting audio to speaker and wobbler appsink.

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

    def enable_wobbling(self, callback: Callable[[SpeechOffsets], None]) -> None:
        """Enable head wobbling driven by audio playback.

        Args:
            callback: Called with ``(x_m, y_m, z_m, roll_rad, pitch_rad,
                yaw_rad)`` for each movement hop.

        """
        if self._head_wobbler is not None:
            self._head_wobbler.stop()
        self._head_wobbler = HeadWobbler(callback, sample_rate=self.WOBBLER_SAMPLE_RATE)
        self._logger.info("Head wobbler enabled (daemon-side)")

    def disable_wobbling(self) -> None:
        """Disable head wobbling."""
        if self._head_wobbler is not None:
            self._head_wobbler.stop()
            self._head_wobbler = None
            self._logger.info("Head wobbler disabled (daemon-side)")

    def set_message_handler(
        self,
        handler: Callable[[str, str], None],  # cb(peer_id, message)
    ) -> None:
        """Set a callback for incoming data channel messages.

        Args:
            handler: Callback function that receives (peer_id, message)

        """
        self._on_data_message = handler

    def set_peer_disconnect_handler(
        self,
        handler: Callable[[str], None],  # cb(peer_id)
    ) -> None:
        """Set a callback fired when a WebRTC peer disconnects.

        The callback runs on the GStreamer/GLib thread (same context as
        ``_consumer_removed``) so consumers must hop back to their own
        loop before touching shared state.
        """
        self._on_peer_disconnect = handler

    def set_session_failed_handler(
        self,
        handler: Callable[[str, str, Dict[str, Any]], None],  # (peer_id, reason, diag)
    ) -> None:
        """Set a callback fired when the negotiation watchdog gives up on a peer.

        The callback runs on the GStreamer/GLib thread (or the
        webrtcbin internal thread for `connection-state == failed`),
        so consumers must hop back to their own loop before doing I/O.
        Typical wiring is to forward to the central signaling relay
        which converts the call into an ``endSession`` message for
        the JS client.

        Args:
            handler: ``(peer_id, reason, diagnostic_dict) -> None``.
                ``reason`` is one of ``SESSION_FAILED_REASON_*``.
                ``diagnostic_dict`` carries the snapshot of the
                webrtcbin state at failure time, suitable for logs.

        """
        self._on_session_failed = handler

    # ------------------------------------------------------------------
    # ICE negotiation watchdog (see `ICE_NEGOTIATION_DEADLINE_S`).
    # ------------------------------------------------------------------

    def _install_negotiation_watchdog(
        self, peer_id: str, webrtcbin: Gst.Element
    ) -> None:
        """Subscribe to webrtcbin's state notifications and start the deadline timer.

        Runs on the GLib main thread (called from `_consumer_added`).
        """
        state = _PeerWebRTCState(peer_id=peer_id)
        with self._peer_states_lock:
            # In theory `consumer-added` fires once per peer_id, but
            # webrtcsink has been seen to re-add a peer after a brief
            # disconnect. Drop the previous watchdog if any to avoid
            # leaking timers.
            existing = self._peer_states.pop(peer_id, None)
            if existing is not None and existing.watchdog_source_id is not None:
                GLib.source_remove(existing.watchdog_source_id)
            self._peer_states[peer_id] = state

        # `notify::*-state` fires every time the named property
        # changes, on whatever thread webrtcbin is using internally.
        # We pass `peer_id` as user data so the handlers don't need
        # to reverse-lookup the peer from the GObject.
        webrtcbin.connect(
            "notify::ice-connection-state",
            self._on_ice_connection_state_change,
            peer_id,
        )
        webrtcbin.connect(
            "notify::connection-state",
            self._on_connection_state_change,
            peer_id,
        )
        webrtcbin.connect(
            "notify::signaling-state",
            self._on_signaling_state_change,
            peer_id,
        )

        source_id = GLib.timeout_add_seconds(
            ICE_NEGOTIATION_DEADLINE_S,
            self._on_negotiation_deadline,
            peer_id,
        )
        with self._peer_states_lock:
            # The peer might have already been removed in the brief
            # window above (rare but possible on flaky networks).
            current = self._peer_states.get(peer_id)
            if current is state:
                current.watchdog_source_id = source_id
            else:
                # Peer is gone; cancel the timer we just scheduled.
                GLib.source_remove(source_id)

    def _teardown_negotiation_watchdog(self, peer_id: str) -> None:
        """Cancel the watchdog timer for `peer_id` and forget its state.

        Called from `_consumer_removed` (peer left cleanly) and from
        `_check_negotiation_deadline` (we just notified failure).
        Safe to call twice — the second call is a no-op.
        """
        with self._peer_states_lock:
            state = self._peer_states.pop(peer_id, None)
        if state is None:
            return
        if state.watchdog_source_id is not None:
            try:
                GLib.source_remove(state.watchdog_source_id)
            except Exception:
                # `source_remove` raises (or returns False, depending
                # on the binding) if the source has already fired.
                # That's fine; we just wanted to make sure it's gone.
                pass

    def _on_ice_connection_state_change(
        self,
        webrtcbin: Gst.Element,
        _pspec: Any,
        peer_id: str,
    ) -> None:
        new_state = self._read_state_nick(webrtcbin, "ice-connection-state")
        with self._peer_states_lock:
            state = self._peer_states.get(peer_id)
            if state is None:
                return
            state.ice_state = new_state
        self._logger.debug(
            f"[watchdog] peer={peer_id} ice-connection-state -> {new_state}"
        )

    def _on_signaling_state_change(
        self,
        webrtcbin: Gst.Element,
        _pspec: Any,
        peer_id: str,
    ) -> None:
        new_state = self._read_state_nick(webrtcbin, "signaling-state")
        with self._peer_states_lock:
            state = self._peer_states.get(peer_id)
            if state is None:
                return
            state.signaling_state = new_state
        self._logger.debug(f"[watchdog] peer={peer_id} signaling-state -> {new_state}")

    def _on_connection_state_change(
        self,
        webrtcbin: Gst.Element,
        _pspec: Any,
        peer_id: str,
    ) -> None:
        new_state = self._read_state_nick(webrtcbin, "connection-state")
        snapshot: Optional[Dict[str, Any]] = None
        should_notify_failure = False
        with self._peer_states_lock:
            state = self._peer_states.get(peer_id)
            if state is None:
                return
            state.conn_state = new_state
            # `failed` is the terminal "we tried and gave up" state
            # webrtcbin reaches when ICE check pairs all fail. We
            # report it eagerly without waiting for the deadline, so
            # the JS client gets a fast rejection on bad networks.
            if new_state == "failed" and not state.failure_notified:
                state.failure_notified = True
                snapshot = state.asdict()
                should_notify_failure = True

        self._logger.info(f"[watchdog] peer={peer_id} connection-state -> {new_state}")

        if should_notify_failure and snapshot is not None:
            self._dispatch_session_failed(
                peer_id,
                SESSION_FAILED_REASON_PC_FAILED,
                snapshot,
            )
            self._teardown_negotiation_watchdog(peer_id)

    def _on_negotiation_deadline(self, peer_id: str) -> bool:
        """Run the watchdog deadline check for ``peer_id``.

        Fired by GLib ``ICE_NEGOTIATION_DEADLINE_S`` seconds after
        ``consumer-added``. Returns False so GLib drops the source
        automatically.
        """
        snapshot: Optional[Dict[str, Any]] = None
        is_stuck = False
        with self._peer_states_lock:
            state = self._peer_states.get(peer_id)
            if state is None:
                # Peer already cleaned up; nothing to do.
                return False
            # Connection is healthy if we're connected or completed.
            # The completed state means ICE has finished checking and
            # promoted a candidate pair.
            if state.conn_state in ("connected",) or state.ice_state in (
                "connected",
                "completed",
            ):
                # All good, but mark the timer as gone so
                # `_teardown_negotiation_watchdog` doesn't try to
                # remove an already-fired source.
                state.watchdog_source_id = None
                return False
            if state.failure_notified:
                # `connection-state == failed` already ran the
                # callback; don't double-fire.
                state.watchdog_source_id = None
                return False
            state.failure_notified = True
            state.watchdog_source_id = None
            snapshot = state.asdict()
            is_stuck = True

        if is_stuck and snapshot is not None:
            self._logger.error(
                f"[watchdog] peer={peer_id} stuck mid-negotiation "
                f"after {ICE_NEGOTIATION_DEADLINE_S}s, snapshot={snapshot}"
            )
            self._dispatch_session_failed(
                peer_id,
                SESSION_FAILED_REASON_ICE_TIMEOUT,
                snapshot,
            )
            self._teardown_negotiation_watchdog(peer_id)
        return False

    def _dispatch_session_failed(
        self,
        peer_id: str,
        reason: str,
        diagnostic: Dict[str, Any],
    ) -> None:
        """Invoke the user-supplied session-failed callback safely.

        Swallows exceptions so a misbehaving handler can't crash the
        GStreamer bus thread.
        """
        handler = self._on_session_failed
        if handler is None:
            self._logger.warning(
                f"[watchdog] peer={peer_id} reason={reason} but no "
                "session-failed handler is wired; the JS client will "
                "have to rely on its own timeout"
            )
            return
        try:
            handler(peer_id, reason, diagnostic)
        except Exception as e:
            self._logger.warning(f"session-failed handler raised for {peer_id}: {e}")

    @staticmethod
    def _read_state_nick(webrtcbin: Gst.Element, prop: str) -> str:
        """Read a webrtcbin enum property as its short string nick.

        Returns ``"connected"`` instead of
        ``GstWebRTCICEConnectionState.connected``. Returns ``"unknown"``
        if introspection fails - we never want a diagnostic helper to raise.
        """
        try:
            value = webrtcbin.get_property(prop)
            nick = getattr(value, "value_nick", None)
            if isinstance(nick, str) and nick:
                return nick
            return str(value)
        except Exception:
            return "unknown"

    def send_data_message(self, peer_id: Optional[str], message: str) -> None:
        """Send a message to connected peers via data channel.

        Args:
            message: The string message to send
            peer_id: If specified, send only to this peer. Otherwise broadcast to all.

        """
        if peer_id:
            if peer_id in self._data_channels:
                self._data_channels[peer_id].emit("send-string", message)
            else:
                self._logger.warning(f"No data channel for peer {peer_id}")
        else:
            # Broadcast to all connected peers
            for channel in self._data_channels.values():
                channel.emit("send-string", message)

    def _setup_data_channel(self, peer_id: str, webrtcbin: Gst.Element) -> None:
        self._logger.debug(f"Setting up data channel for peer {peer_id}")

        # Create data channel options
        options = Gst.Structure.from_string("options,ordered=true")[0]

        # Create the data channel
        channel = webrtcbin.emit("create-data-channel", "data", options)
        if channel:
            self._logger.debug(f"Data channel created for peer {peer_id}")
            self._data_channels[peer_id] = channel

            # Connect to data channel signals
            channel.connect("on-open", self._on_data_channel_open, peer_id)
            channel.connect("on-close", self._on_data_channel_close, peer_id)
            channel.connect("on-message-string", self._on_data_channel_message, peer_id)
            channel.connect("on-error", self._on_data_channel_error, peer_id)
        else:
            self._logger.error(f"Failed to create data channel for peer {peer_id}")

    def _on_data_channel_open(self, channel: Gst.Element, peer_id: str) -> None:
        self._logger.info(f"Data channel opened for peer {peer_id}")

    def _on_data_channel_close(self, channel: Gst.Element, peer_id: str) -> None:
        self._logger.info(f"Data channel closed for peer {peer_id}")
        if peer_id in self._data_channels:
            del self._data_channels[peer_id]

    def _on_data_channel_message(
        self, channel: Gst.Element, message: str, peer_id: str
    ) -> None:
        self._logger.debug(f"Data channel message from peer {peer_id}: {message}")
        if self._on_data_message:
            self._on_data_message(peer_id, message)

    def _on_data_channel_error(
        self, channel: Gst.Element, error: str, peer_id: str
    ) -> None:
        self._logger.error(f"Data channel error for peer {peer_id}: {error}")


if __name__ == "__main__":
    import time

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    server = GstMediaServer(log_level="DEBUG")
    server.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("User interrupted")
    finally:
        server.stop()
        server.close()
