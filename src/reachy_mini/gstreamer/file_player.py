import logging
from threading import Thread
from typing import Optional
import subprocess
from gst_signalling import GstSignallingListener

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
    def __init__(self, log_level: str = "INFO", file : str = "", mode: PlayerMode = PlayerMode.LOCAL, signaling_host: str = "", signaling_port: int = 8443):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)

        if mode == PlayerMode.WEBRTC and signaling_host == "":
            raise ValueError("Signaling host must be set when using WebRTC mode")

        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")

        self._mode = mode
        self._signaling_host = signaling_host
        self._signaling_port = signaling_port

        Gst.init(None)
        self._loop = GLib.MainLoop()
        self._thread_bus_calls: Optional[Thread] = None


        self.pipeline = Gst.Pipeline.new("reachymini_fileplayer")

        self._logger.debug(f"receiving file {file}")

        '''
        filesrc = Gst.ElementFactory.make("filesrc")
        filesrc.set_property("location", file)
        self.pipeline.add(filesrc)

        if mode == PlayerMode.WEBRTC:
            webrtcsink = self._configure_webrtc(self.pipeline, signaling_host, signaling_port)
            filesrc.link(webrtcsink)
        elif mode == PlayerMode.LOCAL:         
            self._configure_audiofile(self.pipeline, filesrc)
        else:
            self._logger.warning("Unknown player mode")
        '''

        self._configure_audiofile(self.pipeline, file)

    def _configure_webrtc(self, pipeline, signaling_host, signaling_port) -> Gst.Element:
        self._logger.debug("Configuring WebRTC")
        webrtcsink = Gst.ElementFactory.make("webrtcsink")
        if not webrtcsink:
            raise RuntimeError("Failed to create webrtcsink element. Is the GStreamer webrtc rust plugin installed?")

        meta_structure = Gst.Structure.new_empty("meta")
        meta_structure.set_value("name", "reachymini_client") # see webrtc_daemon.py
        webrtcsink.set_property("meta", meta_structure)
        signaller = webrtcsink.get_property("signaller")
        signaller.set_property("uri", f"ws://{signaling_host}:{signaling_port}")

        #pipeline.add(webrtcsink)

        return webrtcsink

    def _decodebin_pad_added_cb(self, decodebin, pad):
        self._logger.debug("Decodebin pad added")
        
        audioresample = Gst.ElementFactory.make("audioresample")
        audioconvert = Gst.ElementFactory.make("audioconvert")

        if self._mode == PlayerMode.WEBRTC:
            sink = self._configure_webrtc(self.pipeline, self._signaling_host, self._signaling_port)
        else:
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
        self.pipeline.add(sink)

        pad.link(audioresample.get_static_pad("sink"))
        audioresample.link(audioconvert)
        audioconvert.link(sink)
        audioresample.sync_state_with_parent()
        audioconvert.sync_state_with_parent()
        sink.sync_state_with_parent()


    def _configure_audiofile(self, pipeline, file : str):
        self._logger.debug("Configuring audio")

        filesrc = Gst.ElementFactory.make("filesrc")
        filesrc.set_property("location", file)
        pipeline.add(filesrc)

        decodebin = Gst.ElementFactory.make("decodebin")
        decodebin.connect("pad-added", self._decodebin_pad_added_cb)

        if not all(
            [
                decodebin,
                filesrc
            ]
        ):
            raise RuntimeError("Failed to create GStreamer audio elements")

        pipeline.add(decodebin)
        filesrc.link(decodebin)


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
    import argparse
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    file_path = os.path.join(os.path.dirname(__file__), "../assets/dance1.wav")    
    parser = argparse.ArgumentParser(description="GStreamer File Player")
    parser.add_argument("--mode", choices=["local", "webrtc"], default="local", help="Player mode: local or webrtc")
    parser.add_argument("--file", type=str, default=file_path, help="Path to the audio file")
    parser.add_argument("--signaling_host", type=str, default="127.0.0.1", help="Signaling host for WebRTC")
    parser.add_argument("--signaling_port", type=int, default=8443, help="Signaling port for WebRTC")
    args = parser.parse_args()

    mode = PlayerMode.WEBRTC if args.mode == "webrtc" else PlayerMode.LOCAL
    file_path = args.file
    player = GstFilePlayer(log_level="DEBUG", file=file_path, mode=mode, signaling_host=args.signaling_host, signaling_port=args.signaling_port)
    player.start()
    try: 
        start_time = time.time()
        while time.time() - start_time < 10:
            time.sleep(0.2)
    except KeyboardInterrupt:
        logging.info("User interrupted")
    finally:
        player.stop()