import logging
from threading import Thread
from turtle import mode
from typing import Optional

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import GLib, Gst, GstApp

from reachy_mini.gstreamer.utils import PlayerMode


class GstPlayer:
    def __init__(
        self,
        mode: PlayerMode = PlayerMode.LOCAL,
        signaling_host: str = "",
        signaling_port: int = 8443,
    ):
        self._logger = logging.getLogger(__name__)

        if mode == PlayerMode.WEBRTC and signaling_host == "":
            raise ValueError("Signaling host must be set when using WebRTC mode")

        Gst.init(None)
        self._loop = GLib.MainLoop()
        self._thread_bus_calls: Optional[Thread] = None

        self.pipeline = Gst.Pipeline.new("audio_player")

        # Create elements
        self.appsrc = Gst.ElementFactory.make("appsrc", None)
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("is-live", True)
        caps = Gst.Caps.from_string(
            "audio/x-raw,format=S16LE,channels=1,rate=24000,layout=interleaved"
        )
        self.appsrc.set_property("caps", caps)

        audioconvert = Gst.ElementFactory.make("audioconvert")
        audioresample = Gst.ElementFactory.make("audioresample")

        self.pipeline.add(self.appsrc)
        self.pipeline.add(audioconvert)
        self.pipeline.add(audioresample)

        self.appsrc.link(audioconvert)
        audioconvert.link(audioresample)

        if mode == PlayerMode.WEBRTC:
            webrtcsink = self._configure_webrtc(signaling_host, signaling_port)
            self.pipeline.add(webrtcsink)
            audioresample.link(webrtcsink)
        elif mode == PlayerMode.LOCAL:
            queue = Gst.ElementFactory.make("queue")
            audiosink = Gst.ElementFactory.make("autoaudiosink")  # use default speaker

            self.pipeline.add(queue)
            self.pipeline.add(audiosink)

            self.appsrc.link(queue)
            queue.link(audioconvert)
            audioconvert.link(audioresample)
            audioresample.link(audiosink)
        else:
            self._logger.warning("Unknown player mode")

    def _configure_webrtc(self, signaling_host, signaling_port) -> Gst.Element:
        self._logger.debug("Configuring WebRTC")
        webrtcsink = Gst.ElementFactory.make("webrtcsink")
        if not webrtcsink:
            raise RuntimeError(
                "Failed to create webrtcsink element. Is the GStreamer webrtc rust plugin installed?"
            )

        meta_structure = Gst.Structure.new_empty("meta")
        meta_structure.set_value("name", "reachymini_client")  # see webrtc_daemon.py
        webrtcsink.set_property("meta", meta_structure)
        signaller = webrtcsink.get_property("signaller")
        signaller.set_property("uri", f"ws://{signaling_host}:{signaling_port}")

        return webrtcsink

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

    def _handle_bus_calls(self) -> None:
        self._logger.debug("starting bus message loop")
        bus = self.pipeline.get_bus()
        bus.add_watch(GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop)
        self._loop.run()  # type: ignore[no-untyped-call]
        bus.remove_watch()
        self._logger.debug("bus message loop stopped")

    def play(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        self._thread_bus_calls = Thread(target=self._handle_bus_calls, daemon=True)
        self._thread_bus_calls.start()

    def push_sample(self, data: bytes):
        buf = Gst.Buffer.new_wrapped(data)
        self.appsrc.push_buffer(buf)

    def stop(self):
        self._loop.quit()
        self.pipeline.set_state(Gst.State.NULL)
