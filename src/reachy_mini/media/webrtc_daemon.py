"""WebRTC daemon.

Starts a gstreamer webrtc pipeline to stream video and audio.

This module provides a WebRTC server implementation using GStreamer that can
stream video and audio from the Reachy Mini robot to WebRTC clients. It's
designed to run as a daemon process on the robot and handle multiple client
connections for telepresence and remote monitoring applications.

The WebRTC daemon supports:
- Real-time video streaming from the robot's camera
- Real-time audio streaming from the robot's microphone
- Multiple client connections
- Automatic camera detection and configuration

Example usage:
    >>> from reachy_mini.media.webrtc_daemon import GstWebRTC
    >>>
    >>> # Create and start WebRTC daemon
    >>> webrtc_daemon = GstWebRTC(log_level="INFO")
    >>> # The daemon will automatically start streaming when initialized
    >>>
    >>> # Run until interrupted
    >>> try:
    ...     while True:
    ...         pass  # Keep the daemon running
    ... except KeyboardInterrupt:
    ...     pass  # Cleanup would be handled automatically
"""

import logging
import os
from threading import Thread
from typing import Any, Callable, Dict, Optional, Tuple, cast

import gi

from reachy_mini.daemon.utils import CAMERA_SOCKET_PATH, is_local_camera_available
from reachy_mini.media.camera_constants import (
    ArducamSpecs,
    CameraSpecs,
    ReachyMiniLiteCamSpecs,
    ReachyMiniWirelessCamSpecs,
)

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")

from gi.repository import GLib, Gst  # noqa: E402


class GstWebRTC:
    """WebRTC pipeline using GStreamer.

    This class implements a WebRTC server using GStreamer that streams video
    and audio from the Reachy Mini robot to connected WebRTC clients. It's
    designed to run as a daemon process and handle the complete WebRTC
    signaling and media streaming pipeline.

    Attributes:
        _logger (logging.Logger): Logger instance for WebRTC daemon operations.
        _loop (GLib.MainLoop): GLib main loop for handling GStreamer events.
        camera_specs (CameraSpecs): Specifications of the detected camera.
        _resolution (CameraResolution): Current streaming resolution.
        resized_K (npt.NDArray[np.float64]): Camera intrinsic matrix for current resolution.

    """

    def __init__(
        self,
        log_level: str = "INFO",
    ) -> None:
        """Initialize the GStreamer WebRTC pipeline.

        Args:
            log_level (str): Logging level for WebRTC daemon operations.
                          Default: 'INFO'.

        Note:
            This constructor initializes the GStreamer environment, detects the
            available camera, and sets up the WebRTC streaming pipeline. The
            pipeline automatically starts streaming when initialized.

        Raises:
            RuntimeError: If no camera is detected or camera specifications cannot
                        be determined.

        Example:
            >>> # Initialize WebRTC daemon with debug logging
            >>> webrtc_daemon = GstWebRTC(log_level="DEBUG")
            >>> # The daemon is now streaming and ready to accept client connections

        """
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)

        Gst.init([])
        self._loop = GLib.MainLoop()
        self._thread_bus_calls = Thread(target=lambda: self._loop.run(), daemon=True)
        self._thread_bus_calls.start()

        cam_path, self.camera_specs = self._get_video_device()

        if self.camera_specs is None:
            raise RuntimeError("Camera specs not set")
        self._resolution = self.camera_specs.default_resolution
        self.resized_K = self.camera_specs.K

        if self._resolution is None:
            raise RuntimeError("Failed to get default camera resolution.")

        self._pipeline_sender = Gst.Pipeline.new("reachymini_webrtc_sender")
        self._bus_sender = self._pipeline_sender.get_bus()
        self._bus_sender.add_watch(
            GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop
        )

        webrtcsink = self._configure_webrtc(self._pipeline_sender)

        self._configure_video(cam_path, self._pipeline_sender, webrtcsink)
        self._configure_audio(self._pipeline_sender, webrtcsink)

        self._logger.debug("Configuring data channel")
        self._data_channels: dict[str, Gst.Element] = {}  # peer_id -> channel
        self._on_data_message: Optional[Callable[[str, str], None]] = None

        # Track incoming audio per peer (for bidirectional audio cleanup)
        self._incoming_audio: Dict[str, Dict[str, Any]] = {}

    def __del__(self) -> None:
        """Destructor to ensure gstreamer resources are released."""
        self._logger.debug("Cleaning up GstWebRTC")
        self._loop.quit()
        self._bus_sender.remove_watch()
        # Enable if need to dump logs
        # Gst.deinit()

    def _dump_latency(self) -> None:
        query = Gst.Query.new_latency()
        self._pipeline_sender.query(query)
        self._logger.info(f"Pipeline latency {query.parse_latency()}")

    def _configure_webrtc(self, pipeline: Gst.Pipeline) -> Gst.Element:
        self._logger.debug("Configuring WebRTC")
        webrtcsink = Gst.ElementFactory.make("webrtcsink")
        if not webrtcsink:
            raise RuntimeError(
                "Failed to create webrtcsink element. Is the GStreamer webrtc rust plugin installed?"
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
        """Set media transceivers to sendrecv for bidirectional audio.

        Must be called before the SDP offer is generated (in consumer-added).
        The m-line order can vary (audio/video may swap), so we set all
        transceivers to sendrecv. Video sendrecv is harmless since the
        browser answers recvonly (it has no video track to send).
        """
        for i in range(4):
            try:
                trans = webrtcbin.emit("get-transceiver", i)
                if trans is None:
                    break
                current_dir = trans.get_property("direction")
                self._logger.info(f"Transceiver {i} direction: {current_dir}")
                trans.set_property("direction", self._WEBRTC_DIRECTION_SENDRECV)
                new_dir = trans.get_property("direction")
                self._logger.info(f"Transceiver {i} set to: {new_dir}")
            except Exception:
                break

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

        # Build a separate pipeline: appsrc → depay → decode → convert → speaker
        try:
            playback_pipe = Gst.parse_launch(
                "appsrc name=audio_in format=time is-live=true ! "
                "rtpopusdepay ! opusdec ! audioconvert ! audioresample ! "
                "alsasink device=reachymini_audio_sink sync=false"
            )
        except Exception as e:
            self._logger.error(f"Failed to create audio playback pipeline: {e}")
            return

        appsrc = playback_pipe.get_by_name("audio_in")
        appsrc.set_property("caps", caps)

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
            self._logger.error(
                f"Audio playback error for {peer_id}: {err} {debug}"
            )
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
        self._logger.debug(f"Configuring video {cam_path}")
        camerasrc = Gst.ElementFactory.make("libcamerasrc")
        caps = Gst.Caps.from_string(
            f"video/x-raw,width={self.resolution[0]},height={self.resolution[1]},framerate={self.framerate}/1,format=YUY2,colorimetry=bt709,interlace-mode=progressive"
        )
        capsfilter = Gst.ElementFactory.make("capsfilter")
        capsfilter.set_property("caps", caps)
        tee = Gst.ElementFactory.make("tee")
        # make camera accessible to other applications via unixfdsrc/sink
        unixfdsink = Gst.ElementFactory.make("unixfdsink")
        if is_local_camera_available():
            # prevent crash if socket already exists
            os.remove(CAMERA_SOCKET_PATH)
        unixfdsink.set_property("socket-path", CAMERA_SOCKET_PATH)
        queue_unixfd = Gst.ElementFactory.make("queue", "queue_unixfd")
        queue_encoder = Gst.ElementFactory.make("queue", "queue_encoder")
        v4l2h264enc = Gst.ElementFactory.make("v4l2h264enc")
        extra_controls_structure = Gst.Structure.new_empty("extra-controls")
        # doc: https://docs.qualcomm.com/doc/80-70014-50/topic/v4l2h264enc.html
        extra_controls_structure.set_value("repeat_sequence_header", 1)
        extra_controls_structure.set_value("video_bitrate", 5_000_000)
        extra_controls_structure.set_value("h264_i_frame_period", 60)
        extra_controls_structure.set_value("video_gop_size", 256)
        v4l2h264enc.set_property("extra-controls", extra_controls_structure)
        # Use H264 Level 3.1 + Constrained Baseline for Safari/WebKit compatibility
        caps_h264 = Gst.Caps.from_string(
            "video/x-h264,stream-format=byte-stream,alignment=au,"
            "level=(string)3.1,profile=(string)constrained-baseline"
        )
        capsfilter_h264 = Gst.ElementFactory.make("capsfilter")
        capsfilter_h264.set_property("caps", caps_h264)

        if not all(
            [
                camerasrc,
                capsfilter,
                tee,
                queue_unixfd,
                unixfdsink,
                queue_encoder,
                v4l2h264enc,
                capsfilter_h264,
            ]
        ):
            raise RuntimeError("Failed to create GStreamer video elements")

        pipeline.add(camerasrc)
        pipeline.add(capsfilter)
        pipeline.add(tee)
        pipeline.add(queue_unixfd)
        pipeline.add(unixfdsink)
        pipeline.add(queue_encoder)
        pipeline.add(v4l2h264enc)
        pipeline.add(capsfilter_h264)

        camerasrc.link(capsfilter)
        capsfilter.link(tee)
        tee.link(queue_unixfd)
        queue_unixfd.link(unixfdsink)
        tee.link(queue_encoder)
        queue_encoder.link(v4l2h264enc)
        v4l2h264enc.link(capsfilter_h264)
        capsfilter_h264.link(webrtcsink)

    def _configure_audio(self, pipeline: Gst.Pipeline, webrtcsink: Gst.Element) -> None:
        self._logger.debug("Configuring audio")

        alsasrc = Gst.ElementFactory.make("alsasrc")
        alsasrc.set_property("device", "reachymini_audio_src")
        # to optimize the latency, tune ~/.asoundrc file

        if not all([alsasrc]):
            raise RuntimeError("Failed to create GStreamer audio elements")

        pipeline.add(alsasrc)
        alsasrc.link(webrtcsink)

    def _get_audio_input_device(self) -> Optional[str]:
        """Use Gst.DeviceMonitor to find the pipewire audio card.

        Returns the device ID of the found audio card, None if not.
        """
        monitor = Gst.DeviceMonitor()
        monitor.add_filter("Audio/Source")
        monitor.start()

        snd_card_name = "Reachy Mini Audio"

        devices = monitor.get_devices()
        for device in devices:
            name = device.get_display_name()
            device_props = device.get_properties()

            if snd_card_name in name:
                if device_props and device_props.has_field("object.serial"):
                    serial = device_props.get_string("object.serial")
                    self._logger.debug(f"Found audio input device with serial {serial}")
                    monitor.stop()
                    return str(serial)

        monitor.stop()
        self._logger.warning("No source audio card found.")
        return None

    def _get_video_device(self) -> Tuple[str, Optional[CameraSpecs]]:
        """Use Gst.DeviceMonitor to find the unix camera path /dev/videoX.

        Returns the device path (e.g., '/dev/video2'), or '' if not found.
        """
        monitor = Gst.DeviceMonitor()
        monitor.add_filter("Video/Source")
        monitor.start()

        cam_names = ["Reachy", "Arducam_12MP", "imx708"]

        devices = monitor.get_devices()
        for cam_name in cam_names:
            for device in devices:
                name = device.get_display_name()
                device_props = device.get_properties()

                if cam_name in name:
                    if device_props and device_props.has_field("api.v4l2.path"):
                        device_path = device_props.get_string("api.v4l2.path")
                        camera_specs = (
                            cast(CameraSpecs, ArducamSpecs)
                            if cam_name == "Arducam_12MP"
                            else cast(CameraSpecs, ReachyMiniLiteCamSpecs)
                        )
                        self._logger.debug(f"Found {cam_name} camera at {device_path}")
                        monitor.stop()
                        return str(device_path), camera_specs
                    elif cam_name == "imx708":
                        camera_specs = cast(CameraSpecs, ReachyMiniWirelessCamSpecs)
                        self._logger.debug(f"Found {cam_name} camera")
                        monitor.stop()
                        return cam_name, camera_specs
        monitor.stop()
        self._logger.warning("No camera found.")
        return "", None

    def _on_bus_message(self, bus: Gst.Bus, msg: Gst.Message, loop) -> bool:  # type: ignore[no-untyped-def]
        t = msg.type
        if t == Gst.MessageType.EOS:
            self._logger.warning("End-of-stream")
            return False

        elif t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            self._logger.error(f"Error: {err} {debug}")
            return False

        else:
            # self._logger.warning(f"Unhandled message type: {t}")
            pass

        return True

    def start(self) -> None:
        """Start the WebRTC pipeline."""
        self._logger.debug("Starting WebRTC")
        self._pipeline_sender.set_state(Gst.State.PLAYING)
        GLib.timeout_add_seconds(5, self._dump_latency)

    def pause(self) -> None:
        """Pause the WebRTC pipeline."""
        self._logger.debug("Pausing WebRTC")
        self._pipeline_sender.set_state(Gst.State.PAUSED)

    def stop(self) -> None:
        """Stop the WebRTC pipeline."""
        self._logger.debug("Stopping WebRTC")
        self._pipeline_sender.set_state(Gst.State.NULL)

    # Data channel setup / handling
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

    webrtc = GstWebRTC(log_level="DEBUG")
    webrtc.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("User interrupted")
    finally:
        webrtc.stop()
        webrtc.__del__()
