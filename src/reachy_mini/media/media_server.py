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
from threading import Thread
from typing import Any, Callable, Dict, Optional

import gi

from reachy_mini.daemon.utils import (
    CAMERA_PIPE_NAME,
    CAMERA_SOCKET_PATH,
    is_local_camera_available,
)
from reachy_mini.media.audio_utils import has_reachymini_asoundrc
from reachy_mini.media.camera_constants import (
    CameraSpecs,
    MujocoCameraSpecs,
    ReachyMiniLiteCamSpecs,
)
from reachy_mini.media.device_detection import get_audio_device, get_video_device
from reachy_mini.utils.constants import ASSETS_ROOT_PATH

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")

from gi.repository import GLib, Gst  # noqa: E402


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

    def __init__(
        self,
        log_level: str = "INFO",
        use_sim: bool = False,
    ) -> None:
        """Initialize the GStreamer WebRTC pipeline.

        Args:
            log_level: Logging level for WebRTC daemon operations.
            use_sim: If True, receive video from MuJoCo simulation via UDP
                instead of a physical camera.

        Raises:
            RuntimeError: If no camera is detected (unless use_sim is True)
                or camera specifications cannot be determined.

        """
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)
        self._log_level = log_level
        self._use_sim = use_sim

        Gst.init([])
        self._loop = GLib.MainLoop()
        self._thread_bus_calls = Thread(target=lambda: self._loop.run(), daemon=True)
        self._thread_bus_calls.start()

        if use_sim:
            cam_path = "use_sim"
            self.camera_specs: CameraSpecs = MujocoCameraSpecs()
        else:
            cam_path, detected_specs = get_video_device()
            if detected_specs is None:
                self._logger.warning("No camera found. Video will not be available.")
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
        self._incoming_audio: Dict[str, Dict[str, Any]] = {}
        self._playbin: Optional[Gst.Element] = None

        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Build (or rebuild) the GStreamer pipeline from scratch."""
        self._pipeline_sender = Gst.Pipeline.new("reachymini_webrtc_sender")
        self._bus_sender = self._pipeline_sender.get_bus()
        self._bus_sender.add_watch(
            GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop
        )

        webrtcsink = self._configure_webrtc(self._pipeline_sender)

        self._configure_video(self._cam_path, self._pipeline_sender, webrtcsink)
        self._configure_audio(self._pipeline_sender, webrtcsink)

        self._logger.debug("Pipeline built")

    def __del__(self) -> None:
        """Destructor to ensure gstreamer resources are released."""
        self._logger.debug("Cleaning up GstMediaServer")
        self._loop.quit()
        self._bus_sender.remove_watch()

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

        pipeline.add(webrtcsink)

        return webrtcsink

    def _consumer_added(
        self,
        webrtcsink: Gst.Bin,
        peer_id: str,
        webrtcbin: Gst.Element,
    ) -> None:
        self._logger.info(f"consumer added with peer id: {peer_id}")

        Gst.debug_bin_to_dot_file(
            self._pipeline_sender, Gst.DebugGraphDetails.ALL, "pipeline_full"
        )

        GLib.timeout_add_seconds(5, self._dump_latency)

        self._setup_data_channel(peer_id, webrtcbin)

        # Make audio bidirectional before SDP offer is generated
        self._enable_audio_receive(webrtcbin)

        # Listen for incoming audio pads from the browser (bidirectional audio)
        webrtcbin.connect("pad-added", self._on_consumer_pad_added, peer_id)

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
        playback_pipe = Gst.Pipeline.new(f"audio_playback_{peer_id}")

        appsrc = Gst.ElementFactory.make("appsrc", "audio_in")
        appsrc.set_property("format", Gst.Format.TIME)
        appsrc.set_property("is-live", True)
        appsrc.set_property("caps", caps)

        rtpopusdepay = Gst.ElementFactory.make("rtpopusdepay")
        opusdec = Gst.ElementFactory.make("opusdec")
        audioconvert = Gst.ElementFactory.make("audioconvert")
        audioresample = Gst.ElementFactory.make("audioresample")

        audiosink = self._build_audiosink_element()
        if audiosink is None:
            self._logger.error("Failed to create audio sink element")
            return
        audiosink.set_property("sync", False)

        for elem in [
            appsrc,
            rtpopusdepay,
            opusdec,
            audioconvert,
            audioresample,
            audiosink,
        ]:
            playback_pipe.add(elem)
        appsrc.link(rtpopusdepay)
        rtpopusdepay.link(opusdec)
        opusdec.link(audioconvert)
        audioconvert.link(audioresample)
        audioresample.link(audiosink)

        play_bus = playback_pipe.get_bus()
        play_bus.add_watch(
            GLib.PRIORITY_DEFAULT, self._on_playback_bus_message, peer_id
        )

        playback_pipe.set_state(Gst.State.PLAYING)

        # Pad probe: intercept every RTP buffer, forward to the separate
        # playback pipeline, then DROP so webrtcsink's pipeline is unaffected.
        def _buffer_probe(pad: Gst.Pad, info: Gst.PadProbeInfo, _: None) -> int:
            buf = info.get_buffer()
            if buf is not None:
                appsrc.emit("push-buffer", buf.copy())
            return int(Gst.PadProbeReturn.DROP)

        probe_id = pad.add_probe(Gst.PadProbeType.BUFFER, _buffer_probe, None)

        self._incoming_audio[peer_id] = {
            "playback_pipeline": playback_pipe,
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
        self._logger.info(f"Cleaned up incoming audio for peer {peer_id}")

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
        webrtcsink for WebRTC streaming.
        """
        self._logger.debug("Configuring audio")

        audiosrc = self._build_audio_source()
        if audiosrc is None:
            self._logger.warning("No audio source available. Audio not configured.")
            return

        # Prevent PulseAudio/PipeWire audio sources from becoming the
        # pipeline clock provider.  Their clock causes unixfdsink to stall
        # because it cannot synchronise video buffers against the audio
        # clock.  ALSA sources (wireless CM4) don't have this issue and
        # must keep their default clock behaviour to match the original
        # daemon.  autoaudiosrc is a GstBin and does not expose the
        # property at all.
        factory = audiosrc.get_factory()
        factory_name = factory.get_name() if factory else ""
        if (
            factory_name != "alsasrc"
            and audiosrc.find_property("provide-clock") is not None
        ):
            audiosrc.set_property("provide-clock", False)
            self._logger.debug(f"Set provide-clock=False on {factory_name}")
        else:
            self._logger.debug(
                f"{factory_name} — keeping default provide-clock behaviour."
            )

        pipeline.add(audiosrc)
        audiosrc.link(webrtcsink)

    def _build_audio_source(self) -> Optional[Gst.Element]:
        """Build a platform-aware audio source element.

        Detection order:
        1. .asoundrc — on the wireless CM4 the ``.asoundrc`` file defines
           ALSA aliases (``reachymini_audio_src``).  When present we use
           ``alsasrc`` directly, matching the behaviour of the original
           daemon and avoiding PipeWire clock issues.
        2. GstDeviceMonitor — finds the Reachy Mini Audio card by name and
           returns a platform-native element (pulsesrc, wasapi2src,
           osxaudiosrc).  Used on Lite and desktop platforms.
        3. autoaudiosrc — last resort when no Reachy Mini card is found.

        Returns:
            A GStreamer audio source element, or None if no audio is available.

        """
        # Wireless CM4: .asoundrc defines reachymini_audio_src alias
        if has_reachymini_asoundrc():
            audiosrc = Gst.ElementFactory.make("alsasrc")
            audiosrc.set_property("device", "reachymini_audio_src")
            self._logger.info("Using ALSA device reachymini_audio_src for capture.")
            return audiosrc

        id_audio_card = get_audio_device("Source")

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

    def _on_bus_message(self, bus: Gst.Bus, msg: Gst.Message, loop) -> bool:  # type: ignore[no-untyped-def]
        t = msg.type
        if t == Gst.MessageType.EOS:
            self._logger.warning("End-of-stream")
            return False

        elif t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            self._logger.error(f"Error: {err} {debug}")
            return False

        return True

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

        # Build platform-aware audio sink element
        audiosink = self._build_audiosink_element()

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
        if audiosink is not None:
            playbin.set_property("audio-sink", audiosink)

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
        # Wireless CM4: .asoundrc defines reachymini_audio_sink alias
        if has_reachymini_asoundrc():
            audiosink = Gst.ElementFactory.make("alsasink")
            audiosink.set_property("device", "reachymini_audio_sink")
            self._logger.info("Using ALSA device reachymini_audio_sink for playback.")
            return audiosink

        id_audio_card = get_audio_device("Sink")

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

    def set_message_handler(
        self,
        handler: Callable[[str, str], None],  # cb(peer_id, message)
    ) -> None:
        """Set a callback for incoming data channel messages.

        Args:
            handler: Callback function that receives (peer_id, message)

        """
        self._on_data_message = handler

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
        self._logger.info(f"Data channel message from peer {peer_id}: {message}")
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
        server.__del__()
