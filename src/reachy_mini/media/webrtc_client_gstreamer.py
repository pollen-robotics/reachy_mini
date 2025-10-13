"""GStreamer WebRTC client implementation.

The class is a client for the webrtc server hosted on the Reachy Mini Wireless robot.
"""

from threading import Thread
from typing import Optional

import gi
import numpy as np
import numpy.typing as npt

from reachy_mini.media.audio_base import AudioBase
from reachy_mini.media.camera_base import CameraBase

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import GLib, Gst, GstApp  # noqa: E402, F401


class GstWebRTCClient(CameraBase, AudioBase):
    """GStreamer WebRTC client implementation."""

    SAMPLERATE = 24000

    def __init__(
        self,
        log_level: str = "INFO",
        peer_id: str = "",
        signaling_host: str = "",
        signaling_port: int = 8443,
    ):
        """Initialize the GStreamer WebRTC client."""
        super().__init__(log_level=log_level)
        Gst.init(None)
        self._loop = GLib.MainLoop()
        self._thread_bus_calls: Optional[Thread] = None

        self.pipeline = Gst.Pipeline.new("audio_recorder")

        self._appsink_audio = Gst.ElementFactory.make("appsink")
        caps = Gst.Caps.from_string(
            f"audio/x-raw,channels=1,rate={self.SAMPLERATE},format=S16LE"
        )
        self._appsink_audio.set_property("caps", caps)
        self._appsink_audio.set_property("drop", True)  # avoid overflow
        self._appsink_audio.set_property("max-buffers", 200)
        self.pipeline.add(self._appsink_audio)

        self._appsink_video = Gst.ElementFactory.make("appsink")
        caps_video = Gst.Caps.from_string("video/x-raw,format=BGR")
        self._appsink_video.set_property("caps", caps_video)
        self._appsink_video.set_property("drop", True)  # avoid overflow
        self._appsink_video.set_property("max-buffers", 1)  # keep last image only
        self.pipeline.add(self._appsink_video)

        webrtcsrc = self._configure_webrtcsrc(signaling_host, signaling_port, peer_id)
        self.pipeline.add(webrtcsrc)

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

    def _configure_webrtcbin(self, webrtcsrc: Gst.Element) -> None:
        if isinstance(webrtcsrc, Gst.Bin):
            webrtcbin_name = "webrtcbin0"
            webrtcbin = webrtcsrc.get_by_name(webrtcbin_name)
            assert webrtcbin is not None
            # jitterbuffer has a default 200 ms buffer. Should be ok to lower this in localnetwork config
            webrtcbin.set_property("latency", 50)

    def _webrtcsrc_pad_added_cb(self, webrtcsrc: Gst.Element, pad: Gst.Pad) -> None:
        self._configure_webrtcbin(webrtcsrc)
        if pad.get_name().startswith("video"):
            # the pipeline should be adapted to the app needs

            """
            self._logger.warning("Ignoring video pad")
            sink = Gst.ElementFactory.make("fakesink")
            # sink = Gst.ElementFactory.make("fpsdisplaysink")            
            assert sink is not None
            self.pipeline.add(sink)
            pad.link(sink.get_static_pad("sink"))  # type: ignore[arg-type]
            sink.sync_state_with_parent()
            """

            queue = Gst.ElementFactory.make("queue")

            videoconvert = Gst.ElementFactory.make("videoconvert")
            videoscale = Gst.ElementFactory.make("videoscale")
            videorate = Gst.ElementFactory.make("videorate")

            sink = self._appsink_video
            self.pipeline.add(queue)
            self.pipeline.add(videoconvert)
            self.pipeline.add(videoscale)
            self.pipeline.add(videorate)
            pad.link(queue.get_static_pad("sink"))

            queue.link(videoconvert)
            videoconvert.link(videoscale)
            videoscale.link(videorate)
            videorate.link(sink)

            queue.sync_state_with_parent()
            videoconvert.sync_state_with_parent()
            videoscale.sync_state_with_parent()
            videorate.sync_state_with_parent()
            sink.sync_state_with_parent()

        elif pad.get_name().startswith("audio"):
            pad.link(self._appsink_audio.get_static_pad("sink"))
            self._appsink_audio.sync_state_with_parent()

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

    def _handle_bus_calls(self) -> None:
        self.logger.debug("starting bus message loop")
        bus = self.pipeline.get_bus()
        bus.add_watch(GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop)
        self._loop.run()
        bus.remove_watch()
        self.logger.debug("bus message loop stopped")

    def open(self) -> None:
        """Open the video stream."""
        self.pipeline.set_state(Gst.State.PLAYING)
        self._thread_bus_calls = Thread(target=self._handle_bus_calls, daemon=True)
        self._thread_bus_calls.start()

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

    def get_audio_sample(self) -> Optional[bytes]:
        """Read a sample from the audio card. Returns the sample or None if error.

        Returns:
            Optional[bytes]: The captured sample in raw format, or None if error.

        """
        return self._get_sample(self._appsink_audio)

    def read(self) -> Optional[npt.NDArray[np.uint8]]:
        """Read a frame from the camera. Returns the frame or None if error.

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
        """Stop the pipeline."""
        self._loop.quit()
        self.pipeline.set_state(Gst.State.NULL)

    def get_audio_samplerate(self) -> int:
        """Return the samplerate of the audio device."""
        return self.SAMPLERATE

    def start_recording(self) -> None:
        """Open the audio card using GStreamer."""
        self.logger.warning("start_recording() is not implemented.")

    def stop_recording(self) -> None:
        """Release the camera resource."""
        self.logger.warning("stop_recording() is not implemented.")

    def start_playing(self) -> None:
        """Open the audio output using GStreamer."""
        self.logger.warning("Use open() to start the WebRTC client.")

    def stop_playing(self) -> None:
        """Stop playing audio and release resources."""
        self.logger.warning("Use close() to stop the WebRTC client.")

    def push_audio_sample(self, data: bytes) -> None:
        """Push audio data to the output device."""
        self.logger.warning("Audio playback not implemented in WebRTC client.")

    def play_sound(self, sound_file: str) -> None:
        """Play a sound file.

        Args:
            sound_file (str): Path to the sound file to play.

        """
        self.logger.warning("Audio playback not implemented in WebRTC client.")
