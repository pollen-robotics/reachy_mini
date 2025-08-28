import logging
from threading import Thread
from typing import Optional
import subprocess

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import GLib, Gst, GstApp
import asyncio
from typing import List, Optional, Tuple, Dict
from reachy_mini.gstreamer.utils import PlayerMode
from reachy_mini.gstreamer.utils import get_respeaker_card_number
import os

class GstFilePlayer:
    def __init__(self, log_level: str = "INFO", file : str = ""):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)

        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")
        else:
            self._logger.debug(f"Loading file: {file}")

        Gst.init(None)
        self._loop = GLib.MainLoop()
        self._thread_bus_calls: Optional[Thread] = None

        self.pipeline = Gst.Pipeline.new("reachymini_fileplayer")

        self._configure_audiofile(self.pipeline, file)


    def _decodebin_pad_added_cb(self, decodebin, pad):
        self._logger.debug(f"Decodebin pad added {pad.get_name()}")

        caps = pad.get_current_caps()
        structure = caps.get_structure(0)
        media_type = structure.get_name()

        if not media_type.startswith("audio/"):
            self._logger.warning("This is not an audio pad")
            return

        audioresample = Gst.ElementFactory.make("audioresample")
        audioconvert = Gst.ElementFactory.make("audioconvert")
        # Todo expose this
        volume = Gst.ElementFactory.make("volume")
        volume.set_property("volume", 0.2)

        sink = Gst.ElementFactory.make("alsasink")
        sink.set_property("device", f"hw:{get_respeaker_card_number()}")

        if not all(
            [
                sink,             
                audioresample,
                audioconvert
            ]
        ):
            raise RuntimeError("Failed to create GStreamer audio file elements")

        self.pipeline.add(audioresample)
        self.pipeline.add(audioconvert)
        self.pipeline.add(volume)
        self.pipeline.add(sink)

        pad.link(audioresample.get_static_pad("sink"))        
        audioresample.link(audioconvert)
        audioconvert.link(volume)
        volume.link(sink)

        audioresample.sync_state_with_parent()
        audioconvert.sync_state_with_parent()
        volume.sync_state_with_parent()
        sink.sync_state_with_parent()


    def _configure_audiofile(self, pipeline, file : str):
        self._logger.debug("Configuring audio")


        decodebin = Gst.ElementFactory.make("uridecodebin")
        decodebin.set_property("uri", f"file://{file}")
        decodebin.connect("pad-added", self._decodebin_pad_added_cb)

        if not decodebin:
            raise RuntimeError("Failed to create decodebin")


        pipeline.add(decodebin)

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
        self._logger.debug("Starting Audio Playing")
        self.pipeline.set_state(Gst.State.PLAYING)
        if self._thread_bus_calls is not None:
            self._thread_bus_calls = Thread(target=self._handle_bus_calls, daemon=True)
            self._thread_bus_calls.start()

    def stop(self):
        self._logger.debug("Stopping Audio Playing")
        self._loop.quit()
        self.pipeline.set_state(Gst.State.NULL)

   
if __name__ == "__main__":   
    import time    

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    file_path = os.path.join(os.path.dirname(__file__), "../assets/dance1.wav")    

    file_path = args.file
    player = GstFilePlayer(log_level="DEBUG", file=file_path)
    player.start()
    try: 
        start_time = time.time()
        while time.time() - start_time < 2:
            time.sleep(0.2)
    except KeyboardInterrupt:
        logging.info("User interrupted")
    finally:
        player.stop()