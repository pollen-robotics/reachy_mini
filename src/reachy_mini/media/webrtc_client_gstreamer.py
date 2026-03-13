"""GStreamer WebRTC client — camera + audio over WebRTC.

Connects to the WebRTC server hosted by the Reachy Mini daemon and provides
both video frames and bidirectional audio.  The class exposes the same
public methods as ``GStreamerCamera`` and ``GStreamerAudio`` so that
``MediaManager`` can use it as a drop-in replacement.

Video pipeline (receive)::

    webrtcsrc pad → queue → videoconvert → videoscale → videorate → appsink(BGR)

Audio pipeline (receive)::

    webrtcsrc pad → audioconvert → audioresample → appsink(F32LE)

Audio pipeline (send)::

    appsrc(F32LE) → audioconvert → audioresample → opusenc → rtpopuspay → webrtcbin

Note:
    This class is used internally by ``MediaManager`` when the ``WEBRTC``
    backend is selected.  Direct usage is possible but usually not needed.

Example usage via MediaManager::

    from reachy_mini.media.media_manager import MediaManager, MediaBackend

    media = MediaManager(
        backend=MediaBackend.WEBRTC,
        signalling_host="192.168.1.100",
    )
    frame = media.get_frame()
    media.close()

"""

import logging
from threading import Thread
from typing import Iterator, Optional, cast

try:
    import gi
except ImportError as e:
    raise ImportError(
        "The 'gi' module is required for GstWebRTCClient but could not be imported. "
        "Please check the gstreamer installation."
    ) from e

import numpy as np
import numpy.typing as npt

from reachy_mini.media.audio_doa import AudioDoA
from reachy_mini.media.camera_constants import (
    CameraResolution,
    CameraSpecs,
    MujocoCameraSpecs,
    ReachyMiniLiteCamSpecs,
)
from reachy_mini.media.camera_utils import scale_intrinsics

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import GLib, GObject, Gst, GstApp  # noqa: E402, F401


class GstWebRTCClient:
    """WebRTC client that provides both camera frames and audio.

    Implements the same public API surface as ``GStreamerCamera`` (for
    video) and ``GStreamerAudio`` (for audio) so that ``MediaManager``
    can assign the same instance to both its ``camera`` and ``audio``
    slots.

    Attributes:
        SAMPLE_RATE: Audio sample rate in Hz (16 000).
        CHANNELS: Audio channel count (2 — stereo).
        camera_specs: Camera specifications for resolution / intrinsics.
        resized_K: Intrinsic matrix rescaled to the current resolution.

    """

    SAMPLE_RATE = 16000
    CHANNELS = 2

    def __init__(
        self,
        log_level: str = "INFO",
        peer_id: str = "",
        signaling_host: str = "",
        signaling_port: int = 8443,
        camera_specs: Optional[CameraSpecs] = None,
    ):
        """Initialize the WebRTC client.

        Args:
            log_level: Logging level.
            peer_id: WebRTC peer ID to connect to.
            signaling_host: Host address of the signaling server.
            signaling_port: Port of the signaling server.
            camera_specs: Camera specifications detected by the daemon.
                When ``None`` falls back to ``ReachyMiniLiteCamSpecs``
                with a warning.

        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        Gst.init([])
        self._loop = GLib.MainLoop()
        self._thread_bus_calls = Thread(target=lambda: self._loop.run(), daemon=True)
        self._thread_bus_calls.start()

        self._doa = AudioDoA()

        self._pipeline_record = Gst.Pipeline.new("audio_recorder")
        self._bus_record = self._pipeline_record.get_bus()
        self._bus_record.add_watch(
            GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop
        )

        self._appsink_audio = Gst.ElementFactory.make("appsink")
        caps = Gst.Caps.from_string(
            f"audio/x-raw,rate={self.SAMPLE_RATE},channels={self.CHANNELS},format=F32LE,layout=interleaved"
        )
        self._appsink_audio.set_property("caps", caps)
        self._appsink_audio.set_property("drop", True)  # avoid overflow
        self._appsink_audio.set_property("max-buffers", 500)
        self._pipeline_record.add(self._appsink_audio)

        if camera_specs is not None:
            self.camera_specs: CameraSpecs = camera_specs
        else:
            self.logger.warning(
                "No camera_specs provided — defaulting to ReachyMiniLiteCamSpecs."
            )
            self.camera_specs = cast(CameraSpecs, ReachyMiniLiteCamSpecs)
        self._resolution: Optional[CameraResolution] = None
        self.resized_K: Optional[npt.NDArray[np.float64]] = self.camera_specs.K

        self._appsink_video = Gst.ElementFactory.make("appsink")
        self._appsink_video.set_property("drop", True)  # avoid overflow
        self._appsink_video.set_property("max-buffers", 1)  # keep last image only
        self._pipeline_record.add(self._appsink_video)

        # Set resolution after appsink is created so caps can be properly configured
        self.set_resolution(self.camera_specs.default_resolution)

        self._webrtcsrc = self._configure_webrtcsrc(
            signaling_host, signaling_port, peer_id
        )
        self._pipeline_record.add(self._webrtcsrc)

        self._webrtcbin = None
        self._audio_send_ready = False
        self._appsrc = None
        self._appsrc_pts = 0  # running PTS in nanoseconds for appsrc buffers
        self._playbin: Optional[Gst.Element] = None  # for play_sound
        self._webrtcsrc.connect("deep-element-added", self._on_deep_element_added)
        self.logger.info("GstWebRTCClient initialized (bidirectional audio support)")

    # ------------------------------------------------------------------
    # Resolution / calibration properties
    # ------------------------------------------------------------------

    @property
    def resolution(self) -> tuple[int, int]:
        """Current resolution as ``(width, height)``."""
        if self._resolution is None:
            raise RuntimeError("Camera resolution is not set.")
        return (self._resolution.value[0], self._resolution.value[1])

    @property
    def framerate(self) -> int:
        """Current frame rate in fps."""
        if self._resolution is None:
            raise RuntimeError("Camera resolution is not set.")
        return int(self._resolution.value[2])

    @property
    def K(self) -> Optional[npt.NDArray[np.float64]]:
        """Camera intrinsic matrix for the current resolution, or ``None``."""
        return self.resized_K

    @property
    def D(self) -> Optional[npt.NDArray[np.float64]]:
        """Distortion coefficients, or ``None``."""
        if self.camera_specs is not None:
            return self.camera_specs.D
        return None

    def set_resolution(self, resolution: CameraResolution) -> None:
        """Change the camera resolution.

        Raises:
            RuntimeError: If the pipeline is already playing, if camera
                specs are not set, or for MuJoCo cameras.
            ValueError: If the resolution is not supported.

        """
        if self.camera_specs is None:
            raise RuntimeError(
                "Camera specs not set. Open the camera before setting the resolution."
            )

        if isinstance(self.camera_specs, MujocoCameraSpecs):
            raise RuntimeError(
                "Cannot change resolution of Mujoco simulated camera for now."
            )

        if resolution not in self.camera_specs.available_resolutions:
            raise ValueError(
                f"Resolution not supported. "
                f"Available: {self.camera_specs.available_resolutions}"
            )

        # Rescale intrinsic matrix
        original_K = self.camera_specs.K
        original_size: tuple[int, int] = (
            CameraResolution.R3840x2592at30fps.value[0],
            CameraResolution.R3840x2592at30fps.value[1],
        )
        target_size: tuple[int, int] = (resolution.value[0], resolution.value[1])
        crop_scale = resolution.value[3]
        self.resized_K = scale_intrinsics(
            original_K, original_size, target_size, crop_scale
        )

        # Check pipeline state
        if self._pipeline_record.get_state(0).state == Gst.State.PLAYING:
            raise RuntimeError(
                "Cannot change resolution while the camera is streaming. "
                "Please close the camera first."
            )

        self._resolution = resolution
        caps_video = Gst.Caps.from_string(
            f"video/x-raw,format=BGR,"
            f"width={self._resolution.value[0]},"
            f"height={self._resolution.value[1]},"
            f"framerate={self.framerate}/1"
        )
        self._appsink_video.set_property("caps", caps_video)

    # ------------------------------------------------------------------
    # WebRTC setup
    # ------------------------------------------------------------------

    def _configure_webrtcsrc(
        self, signaling_host: str, signaling_port: int, peer_id: str
    ) -> Gst.Element:
        source = Gst.ElementFactory.make("webrtcsrc")
        if not source:
            raise RuntimeError(
                "Failed to create webrtcsrc element. "
                "Is the GStreamer webrtc rust plugin installed?"
            )

        source.connect("pad-added", self._webrtcsrc_pad_added_cb)
        signaller = source.get_property("signaller")
        signaller.set_property("producer-peer-id", peer_id)
        signaller.set_property("uri", f"ws://{signaling_host}:{signaling_port}")
        return source

    def _on_deep_element_added(
        self, bin: Gst.Bin, sub_bin: Gst.Bin, element: Gst.Element
    ) -> None:
        """Detect the internal webrtcbin element created by webrtcsrc."""
        factory = element.get_factory()
        if factory and factory.get_name() == "webrtcbin":
            self.logger.info(f"Captured webrtcbin: {element.get_name()}")
            self._webrtcbin = element
            element.connect("on-new-transceiver", self._on_new_transceiver)

    def _on_new_transceiver(
        self, webrtcbin: Gst.Element, transceiver: GObject.Object
    ) -> None:
        """Set transceivers to SENDRECV for bidirectional audio.

        When ``codec-preferences`` indicates audio, we set SENDRECV so
        the client can push audio samples back to the daemon.

        When ``codec-preferences`` is absent (video transceiver created
        by webrtcsrc before SDP caps propagate), we also set SENDRECV to
        match the daemon's offer.  This causes webrtcsrc to create an
        internal appsrc for video that emits a non-fatal
        ``not-negotiated`` error — the bus-message handler ignores it.

        Only transceivers explicitly identified as non-audio with known
        caps are left unchanged.
        """
        caps = transceiver.get_property("codec-preferences")
        if caps is not None and caps.get_size() > 0:
            media = caps.get_structure(0).get_string("media")
            if media != "audio":
                return
        transceiver.set_property(
            "direction", 4
        )  # GstWebRTCRTPTransceiverDirection.SENDRECV
        self.logger.info("Transceiver configured for SENDRECV")

    def _dump_latency(self) -> None:
        query = Gst.Query.new_latency()
        self._pipeline_record.query(query)
        self.logger.debug(f"Pipeline latency {query.parse_latency()}")

    def _iterate_gst(self, iterator: Gst.Iterator) -> Iterator[Gst.Element]:
        """Iterate over GStreamer iterators."""
        while True:
            result, elem = iterator.next()
            if result == Gst.IteratorResult.DONE:
                break
            if result == Gst.IteratorResult.OK:
                yield elem
            elif result == Gst.IteratorResult.RESYNC:
                iterator.resync()

    def _configure_webrtcbin(self, webrtcsrc: Gst.Element) -> None:
        if isinstance(webrtcsrc, Gst.Bin):
            webrtcbin = webrtcsrc.get_by_name("webrtcbin0")

            if webrtcbin is None:
                self.logger.debug(
                    f"webrtcbin0 not found, scanning elements in {webrtcsrc.get_name()} recursively..."
                )
                for elem in self._iterate_gst(webrtcsrc.iterate_recurse()):
                    if elem.get_factory().get_name() == "webrtcbin":
                        webrtcbin = elem
                        self.logger.debug(
                            f"Found webrtcbin by factory search: {elem.get_name()}"
                        )
                        break

            assert webrtcbin is not None, (
                "Could not find webrtcbin element in webrtcsrc"
            )
            webrtcbin.set_property("latency", 10)

    def _webrtcsrc_pad_added_cb(self, webrtcsrc: Gst.Element, pad: Gst.Pad) -> None:
        self._configure_webrtcbin(webrtcsrc)
        if pad.get_name().startswith("video"):
            queue = Gst.ElementFactory.make("queue")
            videoconvert = Gst.ElementFactory.make("videoconvert")
            videoscale = Gst.ElementFactory.make("videoscale")
            videorate = Gst.ElementFactory.make("videorate")

            self._pipeline_record.add(queue)
            self._pipeline_record.add(videoconvert)
            self._pipeline_record.add(videoscale)
            self._pipeline_record.add(videorate)
            pad.link(queue.get_static_pad("sink"))

            queue.link(videoconvert)
            videoconvert.link(videoscale)
            videoscale.link(videorate)
            videorate.link(self._appsink_video)

            queue.sync_state_with_parent()
            videoconvert.sync_state_with_parent()
            videoscale.sync_state_with_parent()
            videorate.sync_state_with_parent()
            self._appsink_video.sync_state_with_parent()

        elif pad.get_name().startswith("audio"):
            audioconvert = Gst.ElementFactory.make("audioconvert")
            audioresample = Gst.ElementFactory.make("audioresample")
            self._pipeline_record.add(audioconvert)
            self._pipeline_record.add(audioresample)

            pad.link(audioconvert.get_static_pad("sink"))
            audioconvert.link(audioresample)
            audioresample.link(self._appsink_audio)

            self._appsink_audio.sync_state_with_parent()
            audioconvert.sync_state_with_parent()
            audioresample.sync_state_with_parent()

            # Send path: appsrc → encode → webrtcbin for bidirectional audio
            self._setup_audio_send_chain()

        GLib.timeout_add_seconds(5, self._dump_latency)

    def _on_bus_message(self, bus: Gst.Bus, msg: Gst.Message, loop) -> bool:  # type: ignore[no-untyped-def]
        t = msg.type
        if t == Gst.MessageType.EOS:
            self.logger.warning("End-of-stream")
            return False
        elif t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            src = msg.src

            # webrtcsrc may emit non-fatal errors from its internal
            # elements (e.g. appsrc not-negotiated when a sendrecv
            # transceiver has no data to send).  GStreamer wraps the
            # actual reason as "Internal data stream error." in the
            # GError, with "not-negotiated" only in the debug string.
            # These should not tear down the whole pipeline.
            if (
                src is not None
                and src.get_factory() is not None
                and src.get_factory().get_name() == "appsrc"
                and (
                    "not-negotiated" in str(err)
                    or "Internal data stream error" in str(err)
                )
            ):
                self.logger.debug(f"Ignoring non-fatal webrtcsrc internal error: {err}")
                return True

            self.logger.error(f"Error: {err} {debug}")
            return False
        return True

    # ------------------------------------------------------------------
    # Camera (video) methods
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Start the WebRTC pipeline (both video and audio)."""
        self._pipeline_record.set_state(Gst.State.PLAYING)

    def _get_sample(self, appsink: GstApp.AppSink) -> Optional[bytes]:
        sample = appsink.try_pull_sample(20_000_000)
        if sample is None:
            return None
        data = None
        if isinstance(sample, Gst.Sample):
            buf = sample.get_buffer()
            if buf is None:
                self.logger.warning("Buffer is None")
            data = buf.extract_dup(0, buf.get_size())
        return data

    def read(self) -> Optional[npt.NDArray[np.uint8]]:
        """Pull the latest BGR video frame.

        Returns:
            A NumPy array of shape ``(height, width, 3)`` or ``None``.

        """
        data = self._get_sample(self._appsink_video)
        if data is None:
            return None

        arr = np.frombuffer(data, dtype=np.uint8).reshape(
            (self.resolution[1], self.resolution[0], 3)
        )
        return arr

    def close(self) -> None:
        """Stop the WebRTC pipeline."""
        self._pipeline_record.set_state(Gst.State.NULL)

    # ------------------------------------------------------------------
    # Audio methods
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        """No-op — recording starts automatically with ``open()``."""
        pass

    def get_audio_sample(self) -> Optional[npt.NDArray[np.float32]]:
        """Pull the next recorded audio chunk.

        Returns:
            A float32 array of shape ``(num_samples, 2)`` or ``None``.

        """
        sample = self._get_sample(self._appsink_audio)
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
        """Number of input channels (2)."""
        return self.CHANNELS

    def get_output_channels(self) -> int:
        """Number of output channels (2)."""
        return self.CHANNELS

    def stop_recording(self) -> None:
        """No-op — managed by ``close()``."""
        pass

    def _setup_audio_send_chain(self) -> None:
        """Set up the audio send chain through the existing webrtcbin.

        Builds: appsrc → audioconvert → audioresample → opusenc → rtpopuspay → webrtcbin
        """
        if self._audio_send_ready:
            return
        self._audio_send_ready = True  # prevent re-entry

        self.logger.info("Setting up audio send chain...")
        if self._webrtcbin is None:
            self.logger.error("webrtcbin not found, cannot set up audio send chain")
            self._audio_send_ready = False
            return

        webrtcbin_parent = self._webrtcbin.get_parent()

        # Find the audio sink pad on webrtcbin
        sink_pad = None
        pt = 96
        for pad in self._iterate_gst(self._webrtcbin.iterate_sink_pads()):
            if pad.is_linked():
                continue
            caps = pad.query_caps(None)
            if caps and caps.get_size() > 0:
                s = caps.get_structure(0)
                enc = s.get_string("encoding-name")
                if enc and enc.upper() == "OPUS":
                    sink_pad = pad
                    ok, val = s.get_int("payload")
                    if ok:
                        pt = val
                    self.logger.info(f"Found audio sink pad: {pad.get_name()}, pt={pt}")
                    break

        if sink_pad is None:
            self.logger.error(
                "No OPUS sink pad found on webrtcbin, audio send disabled"
            )
            self._audio_send_ready = False
            return

        appsrc = Gst.ElementFactory.make("appsrc")
        appsrc.set_property("format", Gst.Format.TIME)
        appsrc.set_property("is-live", True)
        caps = Gst.Caps.from_string(
            f"audio/x-raw,format=F32LE,channels={self.CHANNELS},rate={self.SAMPLE_RATE},layout=interleaved"
        )
        appsrc.set_property("caps", caps)

        audioconvert = Gst.ElementFactory.make("audioconvert")
        audioresample = Gst.ElementFactory.make("audioresample")
        opusenc = Gst.ElementFactory.make("opusenc")
        opusenc.set_property("audio-type", "restricted-lowdelay")
        opusenc.set_property("frame-size", 10)
        rtpopuspay = Gst.ElementFactory.make("rtpopuspay")
        rtpopuspay.set_property("pt", pt)

        elems = (appsrc, audioconvert, audioresample, opusenc, rtpopuspay)

        target_bin = webrtcbin_parent if webrtcbin_parent else self._webrtcsrc
        for elem in elems:
            if not target_bin.add(elem):
                self.logger.error(
                    f"Failed to add {elem.get_name()} to {target_bin.get_name()}"
                )
                self._audio_send_ready = False
                return

        appsrc.link(audioconvert)
        audioconvert.link(audioresample)
        audioresample.link(opusenc)
        opusenc.link(rtpopuspay)

        src_pad = rtpopuspay.get_static_pad("src")
        link_result = src_pad.link_full(sink_pad, Gst.PadLinkCheck.NOTHING)
        if link_result != Gst.PadLinkReturn.OK:
            self.logger.error(f"Failed to link rtpopuspay to webrtcbin: {link_result}")
            self._audio_send_ready = False
            return

        for elem in elems:
            elem.sync_state_with_parent()

        self._appsrc = appsrc
        self.logger.info("Audio send chain ready (bidirectional audio enabled)")

    def set_max_output_buffers(self, max_buffers: int) -> None:
        """Limit the number of queued send buffers.

        Args:
            max_buffers: Maximum buffer count.

        """
        if self._appsrc is not None:
            self._appsrc.set_property("max-buffers", max_buffers)
            self._appsrc.set_property("leaky-type", 2)  # drop old buffers
        else:
            self.logger.warning(
                "AppSrc is not initialized. Call start_playing() first."
            )

    def start_playing(self) -> None:
        """No-op — audio send chain is set up automatically on WebRTC connection."""
        pass

    def stop_playing(self) -> None:
        """Reset the PTS counter for the send chain."""
        self._appsrc_pts = 0

    def clear_output_buffer(self) -> None:
        """No-op (WebRTC send chain does not buffer significantly)."""
        pass

    def push_audio_sample(self, data: npt.NDArray[np.float32]) -> None:
        """Push audio data to the remote peer via WebRTC.

        Args:
            data: Float32 audio samples.

        """
        if self._appsrc is None:
            return  # send chain not ready yet, silently drop

        num_samples = data.shape[0]
        duration_ns = (num_samples * Gst.SECOND) // self.SAMPLE_RATE

        buf = Gst.Buffer.new_wrapped(data.tobytes())
        buf.pts = self._appsrc_pts
        buf.duration = duration_ns
        self._appsrc_pts += duration_ns

        self._appsrc.push_buffer(buf)

    def play_sound(self, sound_file: str) -> None:
        """Play a sound file (not supported over WebRTC)."""
        self.logger.warning("Audio playback not implemented in WebRTC client.")

    # ------------------------------------------------------------------
    # Direction of Arrival
    # ------------------------------------------------------------------

    def get_DoA(self) -> tuple[float, bool] | None:
        """Get the Direction of Arrival from the ReSpeaker.

        Returns:
            A tuple ``(angle_radians, speech_detected)`` or ``None``.

        """
        return self._doa.get_DoA()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release all resources."""
        self._doa.close()

    def __del__(self) -> None:
        """Ensure GStreamer resources are released."""
        self.cleanup()
        self._loop.quit()
        self._bus_record.remove_watch()
