import logging
from threading import Thread
from typing import Optional

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import GLib, Gst, GstApp

from reachy_mini.gstreamer.utils import PlayerMode, get_arducam_video_device


class GstRecorder:
    def __init__(
        self,
        mode: PlayerMode,
        peer_id: str = "",
        signaling_host: str = "",
        signaling_port: int = 8443,
    ):
        self._logger = logging.getLogger(__name__)
        Gst.init(None)
        self._loop = GLib.MainLoop()
        self._thread_bus_calls: Optional[Thread] = None

        self.pipeline = Gst.Pipeline.new("audio_recorder")

        self._appsink_audio = Gst.ElementFactory.make("appsink")
        caps = Gst.Caps.from_string("audio/x-raw,channels=1,rate=24000,format=S16LE")
        self._appsink_audio.set_property("caps", caps)
        self._appsink_audio.set_property("drop", True)  # avoid overflow
        self._appsink_audio.set_property("max-buffers", 200)
        self.pipeline.add(self._appsink_audio)

        self._appsink_video = Gst.ElementFactory.make("appsink")
        caps_video = Gst.Caps.from_string(
            "image/jpeg, width=1920, height=1080, framerate=30/1"
        )
        self._appsink_video.set_property("caps", caps_video)
        self._appsink_video.set_property("drop", True)  # avoid overflow
        self._appsink_video.set_property("max-buffers", 1)  # keep last image only
        self.pipeline.add(self._appsink_video)

        if mode == PlayerMode.WEBRTC:
            webrtcsrc = self._configure_webrtcsrc(
                signaling_host, signaling_port, peer_id
            )
            self.pipeline.add(webrtcsrc)

        elif mode == PlayerMode.LOCAL:
            autoaudiosrc = Gst.ElementFactory.make(
                "autoaudiosrc", None
            )  # use default mic
            queue = Gst.ElementFactory.make("queue", None)
            audioconvert = Gst.ElementFactory.make("audioconvert", None)
            audioresample = Gst.ElementFactory.make("audioresample", None)

            if not all(
                [
                    autoaudiosrc,
                    queue,
                    audioconvert,
                    audioresample,
                ]
            ):
                raise RuntimeError("Failed to create GStreamer elements")

            self.pipeline.add(autoaudiosrc)
            self.pipeline.add(queue)
            self.pipeline.add(audioconvert)
            self.pipeline.add(audioresample)

            autoaudiosrc.link(queue)
            queue.link(audioconvert)
            audioconvert.link(audioresample)
            audioresample.link(self._appsink_audio)

            cam_path = get_arducam_video_device()
            if cam_path == "":
                self._logger.warning("Recording pipeline set without camera.")
                self.pipeline.remove(self._appsink_video)
            else:
                camsrc = Gst.ElementFactory.make("v4l2src")
                camsrc.set_property("device", cam_path)
                self.pipeline.add(camsrc)
                camsrc.link(self._appsink_video)

        else:
            self._logger.error("Unsupported player mode")
            raise ValueError("Unsupported player mode")

    def _configure_webrtcsrc(
        self, signaling_host: str, signaling_port: int, peer_id: str
    ):
        source = Gst.ElementFactory.make("webrtcsrc")
        source.connect("pad-added", self.webrtcsrc_pad_added_cb)
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

    def webrtcsrc_pad_added_cb(self, webrtcsrc: Gst.Element, pad: Gst.Pad) -> None:
        self._configure_webrtcbin(webrtcsrc)
        if pad.get_name().startswith("video"):  # type: ignore[union-attr]
            self._logger.warning("Ignoring video pad")
            sink = Gst.ElementFactory.make("fakesink")
            # sink = Gst.ElementFactory.make("fpsdisplaysink")
            assert sink is not None
            self.pipeline.add(sink)
            pad.link(sink.get_static_pad("sink"))  # type: ignore[arg-type]
            sink.sync_state_with_parent()

        elif pad.get_name().startswith("audio"):  # type: ignore[union-attr]
            pad.link(self.appsink_audio.get_static_pad("sink"))  # type: ignore[arg-type]
            self.appsink_audio.sync_state_with_parent()

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

    def record(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        self._thread_bus_calls = Thread(target=self._handle_bus_calls, daemon=True)
        self._thread_bus_calls.start()

    def _get_sample(self, appsink):
        sample = appsink.pull_sample()
        if sample is None:
            return None
        data = None
        if isinstance(sample, Gst.Sample):
            buf = sample.get_buffer()
            if buf is None:
                self._logger.warning("Buffer is None")

            data = buf.extract_dup(0, buf.get_size())
        return data

    def get_audio_sample(self):
        return self._get_sample(self._appsink_audio)

    def get_video_sample(self):
        return self._get_sample(self._appsink_video)

    def stop(self):
        self._loop.quit()
        self.pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO)
    recorder = GstRecorder(mode=PlayerMode.LOCAL)
    recorder.record()
    # Wait for the pipeline to start and capture a frame
    time.sleep(2)
    jpeg_data = recorder.get_video_sample()
    if jpeg_data:
        with open("frame.jpg", "wb") as f:
            f.write(jpeg_data)
        logging.info("Image saved as frame.jpg")
    else:
        logging.error("No image captured")
    recorder.stop()
