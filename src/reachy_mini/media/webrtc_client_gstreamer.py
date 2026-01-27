"""GStreamer WebRTC client implementation.

The class is a client for the webrtc server hosted on the Reachy Mini Wireless robot.

This module provides a GStreamer-based WebRTC client that implements both CameraBase
and AudioBase interfaces, allowing it to be used as a drop-in replacement for
traditional camera and audio devices in the media system.

The WebRTC client supports real-time audio and video streaming over WebRTC protocol,
making it suitable for remote operation and telepresence applications.

Note:
    This class is typically used internally by the MediaManager when the WEBRTC
    backend is selected. Direct usage is possible but usually not necessary.

Example usage via MediaManager:
    >>> from reachy_mini.media.media_manager import MediaManager, MediaBackend
    >>>
    >>> # Create media manager with WebRTC backend
    >>> media = MediaManager(
    ...     backend=MediaBackend.WEBRTC,
    ...     signalling_host="192.168.1.100",
    ...     log_level="INFO"
    ... )
    >>>
    >>> # Use camera functionality
    >>> frame = media.get_frame()
    >>> if frame is not None:
    ...     cv2.imshow("WebRTC Stream", frame)
    ...     cv2.waitKey(1)
    >>>
    >>> # Use audio functionality
    >>> media.start_recording()
    >>> audio_samples = media.get_audio_sample()
    >>>
    >>> # Clean up
    >>> media.close()

"""

from threading import Thread
from typing import Iterator, Optional, cast

import gi
import numpy as np
import numpy.typing as npt

from reachy_mini.media.audio_base import AudioBase
from reachy_mini.media.camera_base import CameraBase
from reachy_mini.media.camera_constants import (
    CameraResolution,
    CameraSpecs,
    ReachyMiniWirelessCamSpecs,
)

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import GLib, Gst, GstApp  # noqa: E402, F401


class GstWebRTCClient(CameraBase, AudioBase):
    """GStreamer WebRTC client implementation.

    This class implements a WebRTC client using GStreamer that can connect to
    a WebRTC server (such as the one hosted on Reachy Mini Wireless) to stream
    audio and video in real-time. It implements both CameraBase and AudioBase
    interfaces, allowing seamless integration with the media system.

    Attributes:
        Inherits all attributes from CameraBase and AudioBase.
        Additionally manages GStreamer pipelines for WebRTC communication.

    """

    def __init__(
        self,
        log_level: str = "INFO",
        peer_id: str = "",
        signaling_host: str = "",
        signaling_port: int = 8443,
    ):
        """Initialize the GStreamer WebRTC client.

        Args:
            log_level (str): Logging level for WebRTC operations.
                          Default: 'INFO'.
            peer_id (str): WebRTC peer ID to connect to. Default: ''.
            signaling_host (str): Host address of the WebRTC signaling server.
                               Default: ''.
            signaling_port (int): Port of the WebRTC signaling server.
                               Default: 8443.

        Note:
            This constructor initializes the GStreamer environment and sets up
            the necessary pipelines for WebRTC communication. The WebRTC connection
            is established when open() is called.

        Example:
            >>> client = GstWebRTCClient(
            ...     peer_id="reachymini",
            ...     signaling_host="192.168.1.100",
            ...     signaling_port=8443
            ... )

        """
        super().__init__(log_level=log_level)
        Gst.init(None)
        self._loop = GLib.MainLoop()
        self._thread_bus_calls = Thread(target=lambda: self._loop.run(), daemon=True)
        self._thread_bus_calls.start()

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

        self.camera_specs = cast(CameraSpecs, ReachyMiniWirelessCamSpecs)

        self._appsink_video = Gst.ElementFactory.make("appsink")
        self._appsink_video.set_property("drop", True)  # avoid overflow
        self._appsink_video.set_property("max-buffers", 1)  # keep last image only
        self._pipeline_record.add(self._appsink_video)

        # Set resolution after appsink is created so caps can be properly configured
        self.set_resolution(self.camera_specs.default_resolution)

        webrtcsrc = self._configure_webrtcsrc(signaling_host, signaling_port, peer_id)
        self._pipeline_record.add(webrtcsrc)

        self._pipeline_playback = Gst.Pipeline.new("audio_player")
        self._init_pipeline_playback(
            self._pipeline_playback, signaling_host, signaling_port
        )
        self._bus_playback = self._pipeline_playback.get_bus()
        self._bus_playback.add_watch(
            GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop
        )

    def __del__(self) -> None:
        """Destructor to ensure gstreamer resources are released."""
        super().__del__()
        self._loop.quit()
        self._bus_record.remove_watch()
        self._bus_playback.remove_watch()

    def set_resolution(self, resolution: CameraResolution) -> None:
        """Set the camera resolution."""
        super().set_resolution(resolution)

        # Check if pipeline is not playing before changing resolution
        if self._pipeline_record.get_state(0).state == Gst.State.PLAYING:
            raise RuntimeError(
                "Cannot change resolution while the camera is streaming. Please close the camera first."
            )

        self._resolution = resolution
        caps_video = Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={self._resolution.value[0]},height={self._resolution.value[1]},framerate={self.framerate}/1"
        )
        self._appsink_video.set_property("caps", caps_video)

    def _configure_webrtcsrc(
        self, signaling_host: str, signaling_port: int, peer_id: str
    ) -> Gst.Element:
        source = Gst.ElementFactory.make("webrtcsrc")
        if not source:
            raise RuntimeError(
                "Failed to create webrtcsrc element. Is the GStreamer webrtc rust plugin installed?"
            )

        source.connect("pad-added", self._webrtcsrc_pad_added_cb)
        signaller = source.get_property("signaller")
        signaller.set_property("producer-peer-id", peer_id)
        signaller.set_property("uri", f"ws://{signaling_host}:{signaling_port}")
        return source

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
            # Try to find by standard name first (fast path)
            webrtcbin = webrtcsrc.get_by_name("webrtcbin0")

            if webrtcbin is None:
                # If not found by name (e.g. re-init scenarios), fallback to recursive search by factory type
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
            # jitterbuffer has a default 200 ms buffer. Should be ok to lower this in localnetwork config
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

        GLib.timeout_add_seconds(5, self._dump_latency)

    def _on_bus_message(self, bus: Gst.Bus, msg: Gst.Message, loop) -> bool:  # type: ignore[no-untyped-def]
        t = msg.type
        if t == Gst.MessageType.EOS:
            self.logger.warning("End-of-stream")
            return False

        elif t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            self.logger.error(f"Error: {err} {debug}")
            return False

        return True

    def open(self) -> None:
        """Open the video stream.

        See CameraBase.open() for complete documentation.
        """
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

    def get_audio_sample(self) -> Optional[npt.NDArray[np.float32]]:
        """Read a sample from the audio card. Returns the sample or None if error.

        See AudioBase.get_audio_sample() for complete documentation.

        Returns:
            Optional[npt.NDArray[np.float32]]: The captured sample in raw format, or None if error.

        """
        sample = self._get_sample(self._appsink_audio)
        if sample is None:
            return None
        return np.frombuffer(sample, dtype=np.float32).reshape(-1, 2)

    def read(self) -> Optional[npt.NDArray[np.uint8]]:
        """Read a frame from the camera. Returns the frame or None if error.

        See CameraBase.read() for complete documentation.

        Returns:
            Optional[npt.NDArray[np.uint8]]: The captured frame in BGR format, or None if error.

        """
        data = self._get_sample(self._appsink_video)
        if data is None:
            return None

        arr = np.frombuffer(data, dtype=np.uint8).reshape(
            (self.resolution[1], self.resolution[0], 3)
        )
        return arr

    def close(self) -> None:
        """Stop the pipeline.

        See CameraBase.close() for complete documentation.
        """
        # self._loop.quit()
        self._pipeline_record.set_state(Gst.State.NULL)

    def start_recording(self) -> None:
        """Open the audio card using GStreamer.

        See AudioBase.start_recording() for complete documentation.
        """
        pass  # already started in open()

    def stop_recording(self) -> None:
        """Release the camera resource.

        See AudioBase.stop_recording() for complete documentation.
        """
        pass  # managed in close()

    def _init_pipeline_playback(
        self, pipeline: Gst.Pipeline, signaling_host: str, signaling_port: int
    ) -> None:
        """Initialize the audio playback pipeline."""
        self._appsrc = Gst.ElementFactory.make("appsrc")
        self._appsrc.set_property("format", Gst.Format.TIME)
        self._appsrc.set_property("is-live", True)
        caps = Gst.Caps.from_string(
            f"audio/x-raw,format=F32LE,channels={self.CHANNELS},rate={self.SAMPLE_RATE},layout=interleaved"
        )
        self._appsrc.set_property("caps", caps)
        audioconvert = Gst.ElementFactory.make("audioconvert")
        audioresample = Gst.ElementFactory.make("audioresample")
        opusenc = Gst.ElementFactory.make("opusenc")
        queue = Gst.ElementFactory.make("queue")
        rtpopuspay = Gst.ElementFactory.make("rtpopuspay")
        udpsink = Gst.ElementFactory.make("udpsink")

        udpsink.set_property("host", signaling_host)
        udpsink.set_property("port", 5000)

        pipeline.add(self._appsrc)
        pipeline.add(audioconvert)
        pipeline.add(audioresample)
        pipeline.add(opusenc)
        pipeline.add(queue)
        pipeline.add(rtpopuspay)
        pipeline.add(udpsink)

        self._appsrc.link(audioconvert)
        audioconvert.link(audioresample)
        audioresample.link(opusenc)
        opusenc.link(queue)
        queue.link(rtpopuspay)
        rtpopuspay.link(udpsink)

    def set_max_output_buffers(self, max_buffers: int) -> None:
        """Set the maximum number of output buffers to queue in the player.

        Args:
            max_buffers (int): Maximum number of buffers to queue.

        """
        if self._appsrc is not None:
            self._appsrc.set_property("max-buffers", max_buffers)
            self._appsrc.set_property("leaky-type", 2)  # drop old buffers
        else:
            self.logger.warning(
                "AppSrc is not initialized. Call start_playing() first."
            )

    def start_playing(self) -> None:
        """Open the audio output using GStreamer.

        See AudioBase.start_playing() for complete documentation.
        """
        self._pipeline_playback.set_state(Gst.State.PLAYING)

    def stop_playing(self) -> None:
        """Stop playing audio and release resources.

        See AudioBase.stop_playing() for complete documentation.
        """
        self._pipeline_playback.set_state(Gst.State.NULL)

    def push_audio_sample(self, data: npt.NDArray[np.float32]) -> None:
        """Push audio data to the output device."""
        if self._appsrc is not None:
            buf = Gst.Buffer.new_wrapped(data.tobytes())
            self._appsrc.push_buffer(buf)

        else:
            self.logger.warning(
                "AppSrc is not initialized. Call start_playing() first."
            )

    def play_sound(self, sound_file: str) -> None:
        """Play a sound file.

        See AudioBase.play_sound() for complete documentation.

        Args:
            sound_file (str): Path to the sound file to play.

        """
        self.logger.warning("Audio playback not implemented in WebRTC client.")
