"""WebRTC daemon.

Starts a gstreamer webrtc pipeline to stream video and audio.
"""

import asyncio
import logging
from threading import Thread
from typing import Dict, List, Optional

import gi
from gst_signalling import GstSignallingListener

from reachy_mini.media.audio_utils import get_respeaker_card_number

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")

from gi.repository import GLib, Gst  # noqa: E402


class GstWebRTC:
    """WebRTC pipeline using GStreamer."""

    def __init__(self, log_level: str = "INFO"):
        """Initialize the GStreamer WebRTC pipeline."""
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)
        Gst.init(None)
        self._loop = GLib.MainLoop()
        self._thread_bus_calls: Optional[Thread] = None

        self._id_audio_card = get_respeaker_card_number()

        self.pipeline = Gst.Pipeline.new("reachymini_webrtc")

        webrtcsink = self._configure_webrtc(self.pipeline)

        self._configure_video(self.pipeline, webrtcsink)
        self._configure_audio(self.pipeline, webrtcsink)

        self._webrtcsrc: Optional[Gst.Element] = None
        self._webrtcsrc_count = -1
        self._peer_audio_id = ""
        self._peer_audio_name = "reachymini_client"
        self._peer_audio_listener: Optional[GstSignallingListener] = None
        self._receiver_elements: List[Gst.Element] = []
        self._asyncio_loop: Optional[asyncio.AbstractEventLoop] = None

        self._thread_webrtcsrc_manager = Thread(
            target=self._webrtcsrc_manager, daemon=True
        )

    def _configure_webrtc(self, pipeline) -> Gst.Element:
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

        pipeline.add(webrtcsink)

        return webrtcsink

    def _configure_video(self, pipeline, webrtcsink):
        self._logger.debug("Configuring video")
        libcamerasrc = Gst.ElementFactory.make("libcamerasrc")
        caps = Gst.Caps.from_string(
            "video/x-raw,width=1280,height=720,framerate=60/1,format=YUY2,colorimetry=bt709,interlace-mode=progressive"
        )
        capsfilter = Gst.ElementFactory.make("capsfilter")
        capsfilter.set_property("caps", caps)
        queue = Gst.ElementFactory.make("queue")
        v4l2h264enc = Gst.ElementFactory.make("v4l2h264enc")
        extra_controls_structure = Gst.Structure.new_empty("extra-controls")
        extra_controls_structure.set_value("repeat_sequence_header", 1)
        v4l2h264enc.set_property("extra-controls", extra_controls_structure)
        caps_h264 = Gst.Caps.from_string(
            "video/x-h264,stream-format=byte-stream,alignment=au,level=(string)4"
        )
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

    def _configure_audio(self, pipeline, webrtcsink):
        self._logger.debug("Configuring audio")
        alsasrc = Gst.ElementFactory.make("alsasrc")
        alsasrc.set_property("device", f"hw:{self._id_audio_card},0")
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
        """Start the WebRTC pipeline."""
        self._logger.debug("Starting WebRTC")
        self.pipeline.set_state(Gst.State.PLAYING)
        self._thread_webrtcsrc_manager.start()
        if self._thread_bus_calls is not None:
            self._thread_bus_calls = Thread(target=self._handle_bus_calls, daemon=True)
            self._thread_bus_calls.start()

    def pause(self):
        """Pause the WebRTC pipeline."""
        self._logger.debug("Pausing WebRTC")
        self.pipeline.set_state(Gst.State.PAUSED)

    def stop(self):
        """Stop the WebRTC pipeline."""
        self._logger.debug("Stopping WebRTC")

        if self._asyncio_loop and self._peer_audio_listener:
            future = asyncio.run_coroutine_threadsafe(
                self._peer_audio_listener.close(), self._asyncio_loop
            )
            future.result()  # Wait for the close coroutine to finish

        self._loop.quit()
        self.pipeline.set_state(Gst.State.NULL)

    def _webrtcsrc_manager(self) -> None:
        # there is asyncio in the daemon for now. we'll limit its usage here.
        self._logger.debug("Starting webrtcsrc manager")
        self._asyncio_loop = asyncio.new_event_loop()
        self._asyncio_loop.run_until_complete(self._webrtcsrc_manager_coroutine())
        self._logger.debug("Stopping webrtcsrc manager")

    async def _webrtcsrc_manager_coroutine(self) -> None:
        self._logger.debug("Starting webrtcsrc manager coroutine")
        self._peer_audio_listener = GstSignallingListener(
            host="127.0.0.1",
            port=8443,
            name="peer_listener",
        )
        self._peer_audio_listener.on(
            "PeerStatusChanged", self._handle_peer_status_changed
        )
        await self._peer_audio_listener.serve4ever()

    def _removing_webrtcsrc(self) -> None:
        self._peer_audio_id = ""
        if self._webrtcsrc is not None:
            self._webrtcsrc.send_event(Gst.Event.new_eos())
            self._webrtcsrc.set_state(Gst.State.NULL)
            self.pipeline.remove(self._webrtcsrc)
            self._webrtcsrc = None
            for elt in self._receiver_elements:
                self.pipeline.remove(elt)
                elt.set_state(Gst.State.NULL)
            self._receiver_elements.clear()
        self._logger.debug("webrtcsrc removed")

    def _configure_webrtcbin(self, webrtcsrc: Gst.Element) -> None:
        if isinstance(webrtcsrc, Gst.Bin):
            webrtcbin_name = "webrtcbin" + str(self._webrtcsrc_count)
            webrtcbin = webrtcsrc.get_by_name(webrtcbin_name)
            assert webrtcbin is not None
            # jitterbuffer has a default 200 ms buffer.
            webrtcbin.set_property("latency", 50)
            self._receiver_elements.append(webrtcbin)

    def _webrtcsrc_pad_added_cb(self, webrtcsrc: Gst.Element, pad: Gst.Pad) -> None:
        if pad is not None and pad.get_name().startswith("audio"):  # type: ignore[union-attr]
            self._logger.info("Connecting audio client")

            self._configure_webrtcbin(webrtcsrc)

            volume = Gst.ElementFactory.make("volume")
            assert volume is not None
            volume.set_property("volume", 0.2)
            sink = Gst.ElementFactory.make("alsasink")
            sink.set_property("device", f"hw:{self._id_audio_card},0")
            assert sink is not None

            self.pipeline.add(volume)
            self.pipeline.add(sink)
            self._receiver_elements.append(volume)
            self._receiver_elements.append(sink)

            volume.link(sink)
            pad.link(volume.get_static_pad("sink"))  # type: ignore[arg-type]

            volume.sync_state_with_parent()
            sink.sync_state_with_parent()
        elif pad.get_name().startswith("video"):
            self._logger.warning("Ignoring video element")
            fakesink = Gst.ElementFactory.make("fakesink")
            assert fakesink is not None
            self.pipeline.add(fakesink)
            fakesink.sync_state_with_parent()
            pad.link(fakesink.get_static_pad("sink"))  # type: ignore[arg-type]
        else:
            self._logger.warning(f"Unhandled pad type: {pad.get_name()}")

    def _add_webrtcsrc(self, peer_audio_id: str) -> Gst.Element:
        webrtcsrc = Gst.ElementFactory.make("webrtcsrc")
        assert webrtcsrc is not None
        self._webrtcsrc_count += 1

        signaller = webrtcsrc.get_property("signaller")
        signaller.set_property("producer-peer-id", peer_audio_id)
        signaller.set_property("uri", "ws://127.0.0.1:8443")

        webrtcsrc.connect("pad-added", self._webrtcsrc_pad_added_cb)

        self.pipeline.add(webrtcsrc)
        webrtcsrc.sync_state_with_parent()
        return webrtcsrc

    async def _adding_webrtcsrc(self, peer_audio_id: str) -> None:
        while True:
            state_change_return, state, pending = self.pipeline.get_state(
                Gst.CLOCK_TIME_NONE
            )
            if (
                state_change_return == Gst.StateChangeReturn.SUCCESS
                and state == Gst.State.PLAYING
            ):
                self._logger.debug("Pipeline is ready and playing")
                break
            elif state_change_return == Gst.StateChangeReturn.FAILURE:
                self._logger.error(
                    "Failed to get the pipeline state, it might be in an error state"
                )
                break
            else:
                self._logger.debug(
                    f"Pipeline is not ready yet, current state: {state.value_nick}"
                )
                await asyncio.sleep(0.5)
        self._webrtcsrc = self._add_webrtcsrc(peer_audio_id)

    async def _handle_peer_status_changed(
        self, peer_id: str, roles: List[str], meta: Dict[str, str]
    ) -> None:
        self._logger.debug(
            f'Peer "{peer_id}" changed roles to {roles} with meta {meta}'
        )
        if meta is None:
            pass
        elif peer_id == self._peer_audio_id and meta["name"] == self._peer_audio_name:
            self._removing_webrtcsrc()
            self._logger.info(f"Client {self._peer_audio_id} is disconnected")
        elif (
            self._peer_audio_id == ""
            and meta["name"] == self._peer_audio_name
            and "producer" in roles
        ):
            self._peer_audio_id = peer_id
            await self._adding_webrtcsrc(self._peer_audio_id)
            self._logger.info(f"Client is connected with id : {self._peer_audio_id}")


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
