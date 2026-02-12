"""Simple gstreamer webrtc producer example.

Sends audio via webrtcsink so the daemon's webrtcsrc receiver can pick it up.
Supports two modes: audiotestsrc (GStreamer built-in) or appsrc (manual push).
"""

import argparse
import time
from threading import Event, Thread

import gi
import numpy as np

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import GLib, Gst, GstApp  # noqa: E402

SAMPLE_RATE = 48000
CHANNELS = 2


class GstSender:
    """Gstreamer webrtc producer that sends audio via webrtcsink."""

    def __init__(
        self,
        signalling_host: str,
        signalling_port: int,
        use_appsrc: bool = False,
    ) -> None:
        """Initialize the sender with signalling server details."""
        Gst.init([])
        self._loop = GLib.MainLoop()
        self._thread_bus_calls = Thread(target=lambda: self._loop.run(), daemon=True)
        self._thread_bus_calls.start()

        self._use_appsrc = use_appsrc
        self._consumer_ready = Event()
        self._appsrc_pts = 0

        self.pipeline = Gst.Pipeline.new("webRTC-sender")

        webrtcsink = Gst.ElementFactory.make("webrtcsink")
        if not webrtcsink:
            print("Failed to create webrtcsink element.")
            exit(-1)

        meta_structure = Gst.Structure.new_empty("meta")
        meta_structure.set_value("name", "reachymini-client")
        webrtcsink.set_property("meta", meta_structure)

        webrtcsink.connect("consumer-added", self._on_consumer_added)

        signaller = webrtcsink.get_property("signaller")
        signaller.set_property("uri", f"ws://{signalling_host}:{signalling_port}")

        if use_appsrc:
            self.appsrc = Gst.ElementFactory.make("appsrc")
            assert self.appsrc is not None
            self.appsrc.set_property("format", Gst.Format.TIME)
            self.appsrc.set_property("is-live", True)
            caps = Gst.Caps.from_string(
                f"audio/x-raw,format=F32LE,channels={CHANNELS},"
                f"rate={SAMPLE_RATE},layout=interleaved"
            )
            self.appsrc.set_property("caps", caps)

            audioconvert = Gst.ElementFactory.make("audioconvert")
            audioresample = Gst.ElementFactory.make("audioresample")

            self.pipeline.add(self.appsrc)
            self.pipeline.add(audioconvert)
            self.pipeline.add(audioresample)
            self.pipeline.add(webrtcsink)

            self.appsrc.link(audioconvert)
            audioconvert.link(audioresample)
            audioresample.link(webrtcsink)
        else:
            self.appsrc = None
            audiotestsrc = Gst.ElementFactory.make("audiotestsrc")
            assert audiotestsrc is not None
            audiotestsrc.set_property("is-live", True)

            self.pipeline.add(audiotestsrc)
            self.pipeline.add(webrtcsink)

            audiotestsrc.link(webrtcsink)

    def _on_consumer_added(
        self, _webrtcsink: Gst.Bin, peer_id: str, _webrtcbin: Gst.Element
    ) -> None:
        """Handle consumer connection."""
        print(f"Consumer added: {peer_id}")
        self._consumer_ready.set()

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
            print("Error starting pipeline.")
            exit(-1)
        mode = "appsrc" if self._use_appsrc else "audiotestsrc"
        print(f"Pipeline playing ({mode}), waiting for consumer ...")

    def stop(self) -> None:
        """Stop the GStreamer pipeline."""
        print("stopping")
        self._consumer_ready.clear()
        self._appsrc_pts = 0
        self.pipeline.send_event(Gst.Event.new_eos())
        self.pipeline.set_state(Gst.State.NULL)

    def push_audio_sample(self, data: np.ndarray) -> None:
        """Push a float32 audio buffer with proper timestamps."""
        if self.appsrc is None:
            return

        num_samples = data.shape[0]
        duration_ns = (num_samples * Gst.SECOND) // SAMPLE_RATE

        buf = Gst.Buffer.new_wrapped(data.tobytes())
        buf.pts = self._appsrc_pts
        buf.duration = duration_ns
        self._appsrc_pts += duration_ns

        self.appsrc.push_buffer(buf)


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
                    print(f"failed to recalculate latency: {e}")
    return True


def audio_push_loop(sender: GstSender, tone_hz: float) -> None:
    """Generate and push a sine wave tone continuously."""
    chunk_duration = 0.02  # 20 ms chunks
    samples_per_chunk = int(SAMPLE_RATE * chunk_duration)
    phase = 0.0

    while True:
        t = np.arange(samples_per_chunk, dtype=np.float32) / SAMPLE_RATE
        mono = 0.5 * np.sin(2.0 * np.pi * tone_hz * t + phase).astype(np.float32)
        phase += 2.0 * np.pi * tone_hz * samples_per_chunk / SAMPLE_RATE
        stereo = np.column_stack((mono, mono)).astype(np.float32)
        sender.push_audio_sample(stereo)
        time.sleep(chunk_duration)


def main() -> None:
    """Run the main function."""
    parser = argparse.ArgumentParser(
        description="webrtc gstreamer producer — sends audio via webrtcsink"
    )
    parser.add_argument(
        "--signaling-host",
        default="127.0.0.1",
        help="Gstreamer signaling host - Reachy Mini ip",
    )
    parser.add_argument(
        "--signaling-port", default=8443, type=int, help="Gstreamer signaling port"
    )
    parser.add_argument(
        "--source",
        choices=["appsrc", "audiotestsrc"],
        default="audiotestsrc",
        help="Audio source type (default: audiotestsrc)",
    )
    parser.add_argument(
        "--tone-hz",
        default=440.0,
        type=float,
        help="Sine wave frequency in Hz (appsrc mode only)",
    )
    args = parser.parse_args()

    use_appsrc = args.source == "appsrc"
    sender = GstSender(args.signaling_host, args.signaling_port, use_appsrc=use_appsrc)
    sender.play()

    if use_appsrc:
        t = Thread(target=lambda: audio_push_loop(sender, args.tone_hz), daemon=True)
        t.start()

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
