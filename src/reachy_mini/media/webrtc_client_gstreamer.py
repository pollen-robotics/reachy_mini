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
            f"audio/x-raw,channels=2,rate={self.SAMPLE_RATE},format=F32LE"
        )
        self._appsink_audio.set_property("caps", caps)
        self._appsink_audio.set_property("drop", True)  # avoid overflow
        self._appsink_audio.set_property("max-buffers", 500)
        self.pipeline.add(self._appsink_audio)

        self._appsink_video = Gst.ElementFactory.make("appsink")
        caps_video = Gst.Caps.from_string("video/x-raw,format=BGR")
        self._appsink_video.set_property("caps", caps_video)
        self._appsink_video.set_property("drop", True)  # avoid overflow
        self._appsink_video.set_property("max-buffers", 1)  # keep last image only
        self.pipeline.add(self._appsink_video)

        self._signaling_host = signaling_host
        self._signaling_port = signaling_port

        webrtcsrc = self._configure_webrtcsrc(signaling_host, signaling_port, peer_id)
        self.pipeline.add(webrtcsrc)

        self._webrtcsink: Optional[Gst.Element] = None
        self._appsrc: Optional[Gst.Element] = None

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
            queue = Gst.ElementFactory.make("queue")

            videoconvert = Gst.ElementFactory.make("videoconvert")
            videoscale = Gst.ElementFactory.make("videoscale")
            videorate = Gst.ElementFactory.make("videorate")

            self.pipeline.add(queue)
            self.pipeline.add(videoconvert)
            self.pipeline.add(videoscale)
            self.pipeline.add(videorate)
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
            self.pipeline.add(audioconvert)
            self.pipeline.add(audioresample)

            pad.link(audioconvert.get_static_pad("sink"))
            audioconvert.link(audioresample)
            audioresample.link(self._appsink_audio)

            self._appsink_audio.sync_state_with_parent()
            audioconvert.sync_state_with_parent()
            audioresample.sync_state_with_parent()

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

    def get_audio_sample(self) -> Optional[npt.NDArray[np.float32]]:
        """Read a sample from the audio card. Returns the sample or None if error.

        Returns:
            Optional[npt.NDArray[np.float32]]: The captured sample in raw format, or None if error.

        """
        sample = self._get_sample(self._appsink_audio)
        if sample is None:
            return None
        return np.frombuffer(sample, dtype=np.float32).reshape(-1, 2)

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
        pass  # already started in open()

    def stop_recording(self) -> None:
        """Release the camera resource."""
        pass  # managed in close()

    def start_playing(self) -> None:
        """Open the audio output using GStreamer."""
        if self._appsrc is not None and self._webrtcsink is not None:
            self.logger.warning("Audio playback already started.")
            return

        self._webrtcsink = Gst.ElementFactory.make("webrtcsink")
        signaller = self._webrtcsink.get_property("signaller")
        signaller.set_property(
            "uri", f"ws://{self._signaling_host}:{self._signaling_port}"
        )
        meta_structure = Gst.Structure.new_empty("meta")
        meta_structure.set_value("name", "reachymini_client")
        self._webrtcsink.set_property("meta", meta_structure)

        self._appsrc = Gst.ElementFactory.make("appsrc")
        self._appsrc.set_property("format", Gst.Format.TIME)
        self._appsrc.set_property("is-live", True)
        caps = Gst.Caps.from_string(
            f"audio/x-raw,format=F32LE,channels=1,rate={self.SAMPLE_RATE},layout=interleaved"
        )
        self._appsrc.set_property("caps", caps)

        self.pipeline.add(self._appsrc)
        self.pipeline.add(self._webrtcsink)
        self._appsrc.link(self._webrtcsink)
        self._webrtcsink.sync_state_with_parent()
        self._appsrc.sync_state_with_parent()

    def stop_playing(self) -> None:
        """Stop playing audio and release resources."""
        if self._appsrc is not None and self._webrtcsink is not None:
            self._appsrc.set_state(Gst.State.NULL)
            self.pipeline.remove(self._appsrc)
            self._webrtcsink.send_event(Gst.Event.new_eos())
            self._webrtcsink.set_state(Gst.State.NULL)
            self.pipeline.remove(self._webrtcsink)

    def push_audio_sample(self, data: bytes) -> None:
        """Push audio data to the output device."""
        if self._appsrc is not None:
            buf = Gst.Buffer.new_wrapped(data)
            self._appsrc.push_buffer(buf)
        else:
            self.logger.warning(
                "AppSrc is not initialized. Call start_playing() first."
            )

    def play_sound(self, sound_file: str) -> None:
        """Play a sound file.

        Args:
            sound_file (str): Path to the sound file to play.

        """
        self.logger.warning("Audio playback not implemented in WebRTC client.")
