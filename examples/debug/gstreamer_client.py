"""Simple gstreamer webrtc consumer example."""

import argparse
from threading import Thread

import gi
from gst_signalling.utils import find_producer_peer_id_by_name

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402


class GstConsumer:
    """Gstreamer webrtc consumer class."""

    def __init__(
        self,
        signalling_host: str,
        signalling_port: int,
        peer_name: str,
    ) -> None:
        """Initialize the consumer with signalling server details and peer name."""
        Gst.init(None)
        self._loop = GLib.MainLoop()
        self._thread_bus_calls = Thread(target=lambda: self._loop.run(), daemon=True)
        self._thread_bus_calls.start()

        self.pipeline = Gst.Pipeline.new("webRTC-consumer")
        self.source = Gst.ElementFactory.make("webrtcsrc")

        if not self.pipeline:
            print("Pipeline could not be created.")
            exit(-1)

        if not self.source:
            print(
                "webrtcsrc component could not be created. Please make sure that the plugin is installed \
                (see https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs/-/tree/main/net/webrtc)"
            )
            exit(-1)

        self.pipeline.add(self.source)

        peer_id = find_producer_peer_id_by_name(
            signalling_host, signalling_port, peer_name
        )
        print(f"found peer id: {peer_id}")

        self.source.connect("pad-added", self.webrtcsrc_pad_added_cb)
        signaller = self.source.get_property("signaller")
        signaller.set_property("producer-peer-id", peer_id)
        signaller.set_property("uri", f"ws://{signalling_host}:{signalling_port}")

    def dump_latency(self) -> None:
        """Dump the current pipeline latency."""
        query = Gst.Query.new_latency()
        self.pipeline.query(query)
        print(f"Pipeline latency {query.parse_latency()}")

    def _configure_webrtcbin(self, webrtcsrc: Gst.Element) -> None:
        if isinstance(webrtcsrc, Gst.Bin):
            webrtcbin_name = "webrtcbin0"
            webrtcbin = webrtcsrc.get_by_name(webrtcbin_name)
            assert webrtcbin is not None
            # jitterbuffer has a default 200 ms buffer.
            webrtcbin.set_property("latency", 50)

    def webrtcsrc_pad_added_cb(self, webrtcsrc: Gst.Element, pad: Gst.Pad) -> None:
        """Add webrtcsrc elements when a new pad is added."""
        self._configure_webrtcbin(webrtcsrc)
        if pad.get_name().startswith("video"):  # type: ignore[union-attr]
            # webrtcsrc automatically decodes and convert the video
            sink = Gst.ElementFactory.make("fpsdisplaysink")
            assert sink is not None
            self.pipeline.add(sink)
            pad.link(sink.get_static_pad("sink"))  # type: ignore[arg-type]
            sink.sync_state_with_parent()

        elif pad.get_name().startswith("audio"):  # type: ignore[union-attr]
            # webrtcsrc automatically decodes and convert the audio
            sink = Gst.ElementFactory.make("autoaudiosink")
            assert sink is not None
            self.pipeline.add(sink)
            pad.link(sink.get_static_pad("sink"))  # type: ignore[arg-type]
            sink.sync_state_with_parent()

        GLib.timeout_add_seconds(5, self.dump_latency)

    def __del__(self) -> None:
        """Destructor to clean up GStreamer resources."""
        self._loop.quit()
        Gst.deinit()

    def get_bus(self) -> Gst.Bus:
        """Get the GStreamer bus for the pipeline."""
        return self.pipeline.get_bus()

    def play(self) -> None:
        """Start the GStreamer pipeline."""
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Error starting playback.")
            exit(-1)
        print("playing ... (ctrl+c to quit)")

    def stop(self) -> None:
        """Stop the GStreamer pipeline."""
        print("stopping")
        self.pipeline.send_event(Gst.Event.new_eos())
        self.pipeline.set_state(Gst.State.NULL)


def process_msg(bus: Gst.Bus, pipeline: Gst.Pipeline) -> bool:
    """Process messages from the GStreamer bus."""
    msg = bus.timed_pop_filtered(10 * Gst.MSECOND, Gst.MessageType.ANY)
    if msg:
        if msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print(f"Error: {err}, {debug}")
            return False
        elif msg.type == Gst.MessageType.EOS:
            print("End-Of-Stream reached.")
            return False
        elif msg.type == Gst.MessageType.LATENCY:
            if pipeline:
                try:
                    pipeline.recalculate_latency()
                except Exception as e:
                    print("failed to recalculate latency, exception: %s" % str(e))
        # else:
        #    print(f"Message: {msg.type}")
    return True


def main() -> None:
    """Run the main function."""
    parser = argparse.ArgumentParser(description="webrtc gstreamer simple consumer")
    parser.add_argument(
        "--signaling-host",
        default="127.0.0.1",
        help="Gstreamer signaling host - Reachy Mini ip",
    )
    parser.add_argument(
        "--signaling-port", default=8443, help="Gstreamer signaling port"
    )

    args = parser.parse_args()

    consumer = GstConsumer(
        args.signaling_host,
        args.signaling_port,
        "reachymini",
    )
    consumer.play()

    # Wait until error or EOS
    bus = consumer.get_bus()
    try:
        while True:
            if not process_msg(bus, consumer.pipeline):
                break

    except KeyboardInterrupt:
        print("User exit")
    finally:
        consumer.stop()


if __name__ == "__main__":
    main()
