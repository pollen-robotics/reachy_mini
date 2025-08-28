import argparse

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst


class GstSender:
    def __init__(
        self,
        signalling_host: str,
        signalling_port: int,
        peer_name: str,
    ) -> None:
        Gst.init(None)

        self.pipeline = Gst.Pipeline.new("webRTC-sender")
        autoaudiosrc = Gst.ElementFactory.make("audiotestsrc")
        autoaudiosrc.set_property("is-live", True)
        autoaudiosrc.set_property("wave", "pink-noise")
        sink = Gst.ElementFactory.make("webrtcsink")

        if not self.pipeline:
            print("Pipeline could be created.")
            exit(-1)

        if not sink:
            print(
                "webrtcsink component could not be created. Please make sure that the plugin is installed \
                (see https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs/-/tree/main/net/webrtc)"
            )
            exit(-1)

        self.pipeline.add(autoaudiosrc)
        self.pipeline.add(sink)
        autoaudiosrc.link(sink)

        meta_structure = Gst.Structure.new_empty("meta")
        meta_structure.set_value("name", "reachymini_client")  # see webrtc_daemon.py
        sink.set_property("meta", meta_structure)
        signaller = sink.get_property("signaller")
        signaller.set_property("uri", f"ws://{signalling_host}:{signalling_port}")

    def dump_latency(self) -> None:
        query = Gst.Query.new_latency()
        self.pipeline.query(query)
        print(f"Pipeline latency {query.parse_latency()}")

    def get_bus(self) -> Gst.Bus:
        return self.pipeline.get_bus()

    def play(self) -> None:
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Error starting playback.")
            exit(-1)
        print("playing ... (ctrl+c to quit)")

    def stop(self) -> None:
        print("stopping")
        self.pipeline.send_event(Gst.Event.new_eos())
        self.pipeline.set_state(Gst.State.NULL)


def process_msg(bus: Gst.Bus, pipeline: Gst.Pipeline) -> bool:
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
                    print("failed to recalculate warning, exception: %s" % str(e))
        # else:
        #    print(f"Message: {msg.type}")
    return True


def main() -> None:
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

    sender = GstSender(
        args.signaling_host,
        args.signaling_port,
        "reachymini_client",
    )
    sender.play()

    # Wait until error or EOS
    bus = sender.get_bus()
    try:
        while True:
            if not process_msg(bus, sender.pipeline):
                break

    except KeyboardInterrupt:
        print("User exit")
    finally:
        sender.stop()


if __name__ == "__main__":
    main()
