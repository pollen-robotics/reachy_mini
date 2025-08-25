import logging
from threading import Thread
from typing import Optional
import subprocess

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import GLib, Gst, GstApp


class GstWebRTC:
    def __init__(self, log_level: str = "INFO"):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)
        Gst.init(None)
        self._loop = GLib.MainLoop()
        self._thread_bus_calls: Optional[Thread] = None

        self.pipeline = Gst.Pipeline.new("reachymini_webrtc")

        webrtcsink = self._configure_webrtc(self.pipeline)

        self._configure_video(self.pipeline, webrtcsink)
        self._configure_audio(self.pipeline, webrtcsink)

    def _configure_webrtc(self, pipeline) -> Gst.Element:
        self._logger.debug("Configuring WebRTC")
        webrtcsink = Gst.ElementFactory.make("webrtcsink")
        if not webrtcsink:
            raise RuntimeError("Failed to create webrtcsink element. Is the GStreamer webrtc rust plugin installed?")

        meta_structure = Gst.Structure.new_empty("meta")
        meta_structure.set_value("name", "reachymini")
        webrtcsink.set_property("meta", meta_structure)
        webrtcsink.set_property("run-signalling-server", True)

        pipeline.add(webrtcsink)

        return webrtcsink

    def _configure_video(self, pipeline, webrtcsink):  
        self._logger.debug("Configuring video")
        libcamerasrc = Gst.ElementFactory.make("libcamerasrc")
        caps = Gst.Caps.from_string("video/x-raw,width=1280,height=720,framerate=60/1,format=YUY2,colorimetry=bt709,interlace-mode=progressive")
        capsfilter = Gst.ElementFactory.make("capsfilter")
        capsfilter.set_property("caps", caps)        
        queue = Gst.ElementFactory.make("queue")
        v4l2h264enc = Gst.ElementFactory.make("v4l2h264enc")
        extra_controls_structure = Gst.Structure.new_empty("extra-controls")
        extra_controls_structure.set_value("repeat_sequence_header", 1)
        v4l2h264enc.set_property("extra-controls", extra_controls_structure)
        caps_h264 = Gst.Caps.from_string("video/x-h264,stream-format=byte-stream,alignment=au,level=(string)4")
        capsfilter_h264 = Gst.ElementFactory.make("capsfilter")
        capsfilter_h264.set_property("caps", caps_h264) 
        h264parse = Gst.ElementFactory.make("h264parse")

        if not all(
            [
                libcamerasrc,
                capsfilter,
                queue,
                v4l2h264enc,
                capsfilter_h264,
                h264parse,
            ]
        ):
            raise RuntimeError("Failed to create GStreamer video elements")

        pipeline.add(libcamerasrc)
        pipeline.add(capsfilter)
        pipeline.add(queue)
        pipeline.add(v4l2h264enc)
        pipeline.add(capsfilter_h264)

        libcamerasrc.link(capsfilter)
        capsfilter.link(queue)
        queue.link(v4l2h264enc)
        v4l2h264enc.link(capsfilter_h264)
        capsfilter_h264.link(webrtcsink)

    def _get_respeaker_card_number(self) -> int:
        try:
            result = subprocess.run(['arecord', '-l'], capture_output=True, text=True, check=True)
            output = result.stdout

            lines = output.split('\n')
            for line in lines:
                if 'ReSpeaker' in line and 'card' in line:
                    card_number = line.split(':')[0].split('card ')[1].strip()
                    self._logger.debug(f"Found ReSpeaker sound card: {card_number}")
                    return int(card_number)

            self._logger.warning("ReSpeaker sound card not found. Returning default card")
            return 0  # default sound card

        except subprocess.CalledProcessError as e:
            self._logger.error(f"Cannot find sound card: {e}")
            return 0

    def _configure_audio(self, pipeline, webrtcsink):
        self._logger.debug("Configuring audio")
        alsasrc = Gst.ElementFactory.make("alsasrc")
        id_card = self._get_respeaker_card_number()
        alsasrc.set_property("device", f"hw:{id_card},0")
        queue = Gst.ElementFactory.make("queue")
        audioconvert = Gst.ElementFactory.make("audioconvert")
        audioresample = Gst.ElementFactory.make("audioresample")
        opusenc = Gst.ElementFactory.make("opusenc")
        caps = Gst.Caps.from_string("audio/x-opus,channels=1,rate=48000")
        capsfilter = Gst.ElementFactory.make("capsfilter")
        capsfilter.set_property("caps", caps)

        if not all(
            [
                alsasrc,
                queue,
                audioconvert,
                audioresample,
                opusenc,
                capsfilter,
            ]
        ):
            raise RuntimeError("Failed to create GStreamer audio elements")

        pipeline.add(alsasrc)
        pipeline.add(queue)
        pipeline.add(audioconvert)
        pipeline.add(audioresample)
        pipeline.add(opusenc)
        pipeline.add(capsfilter)

        alsasrc.link(queue)
        queue.link(audioconvert)
        audioconvert.link(audioresample)
        audioresample.link(opusenc)
        opusenc.link(capsfilter)
        capsfilter.link(webrtcsink)

    def _on_bus_message(self, bus: Gst.Bus, msg: Gst.Message, loop) -> bool:  # type: ignore[no-untyped-def]
        t = msg.type
        if t == Gst.MessageType.EOS:
            self._logger.warning("End-of-stream")
            return False

        elif t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            self._logger.error(f"Error: {err} {debug}")
            return False

        elif t == Gst.MessageType.LOST_CLOCK:
            self._logger.warning("Clock lost")
            return False

        else:
            self._logger.warning(f"Unhandled message type: {t}")

        return True

    def _handle_bus_calls(self) -> None:
        self._logger.debug("starting bus message loop")
        bus = self.pipeline.get_bus()
        bus.add_watch(GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop)
        self._loop.run()  # type: ignore[no-untyped-call]
        bus.remove_watch()
        self._logger.debug("bus message loop stopped")

    def start(self):
        self._logger.debug("Starting WebRTC")
        self.pipeline.set_state(Gst.State.PLAYING)
        if self._thread_bus_calls is not None:
            self._thread_bus_calls = Thread(target=self._handle_bus_calls, daemon=True)
            self._thread_bus_calls.start()

    def stop(self):
        self._logger.debug("Stopping WebRTC")
        self._loop.quit()
        self.pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    import time
    webrtc = GstWebRTC(log_level="DEBUG")
    webrtc.start()
    try: 
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("User interrupted")
    finally:
        webrtc.stop()