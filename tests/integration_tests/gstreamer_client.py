"""Integration test for GStreamer media clients.

Tests both client paths:
- **webrtc**: connects via WebRTC (remote or local signalling server)
- **ipc**: reads raw BGR frames from the daemon's unix-socket IPC endpoint

Can optionally start a local GstMediaServer (--start-server) so the test
works on a Lite setup without a running daemon.

Usage examples::

    # WebRTC client, daemon already running:
    python gstreamer_client.py --mode webrtc

    # IPC client, daemon already running:
    python gstreamer_client.py --mode ipc

    # Start server + WebRTC client (Lite, no daemon):
    python gstreamer_client.py --mode webrtc --start-server

    # Start server + IPC client (Lite, no daemon):
    python gstreamer_client.py --mode ipc --start-server

    # Start server with simulated video:
    python gstreamer_client.py --mode ipc --start-server --sim
"""

import argparse
import logging
import platform
import time
from threading import Thread
import gi

from reachy_mini.daemon.utils import CAMERA_PIPE_NAME, CAMERA_SOCKET_PATH
from reachy_mini.media.webrtc_utils import find_producer_peer_id_by_name

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402


class IpcConsumer:
    """GStreamer IPC consumer — reads raw BGR frames from the daemon's unix-socket endpoint."""

    def __init__(self) -> None:
        """Initialize the IPC consumer pipeline: unixfdsrc → queue → videoconvert → fpsdisplaysink."""
        Gst.init([])
        self._loop = GLib.MainLoop()
        self._thread_bus_calls = Thread(target=lambda: self._loop.run(), daemon=True)
        self._thread_bus_calls.start()

        self.pipeline = Gst.Pipeline.new("ipc-consumer")
        if not self.pipeline:
            print("Pipeline could not be created.")
            exit(-1)

        # IPC source (platform-specific)
        if platform.system() == "Windows":
            src = Gst.ElementFactory.make("win32ipcvideosrc")
            if not src:
                print("win32ipcvideosrc could not be created.")
                exit(-1)
            src.set_property("pipe-name", CAMERA_PIPE_NAME)
        else:
            src = Gst.ElementFactory.make("unixfdsrc")
            if not src:
                print(
                    "unixfdsrc could not be created. "
                    "Is the unixfd GStreamer plugin installed?"
                )
                exit(-1)
            src.set_property("socket-path", CAMERA_SOCKET_PATH)

        queue = Gst.ElementFactory.make("queue")
        convert = Gst.ElementFactory.make("videoconvert")
        sink = Gst.ElementFactory.make("fpsdisplaysink")

        if not all([queue, convert, sink]):
            print("Failed to create one or more GStreamer elements.")
            exit(-1)

        self.pipeline.add(src)
        self.pipeline.add(queue)
        self.pipeline.add(convert)
        self.pipeline.add(sink)

        src.link(queue)
        queue.link(convert)
        convert.link(sink)

        bus = self.pipeline.get_bus()
        bus.add_watch(GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop)

    def _on_bus_message(
        self, _bus: Gst.Bus, msg: Gst.Message, _loop: GLib.MainLoop
    ) -> bool:
        """Handle bus messages."""
        if msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print(f"Error: {err}, {debug}")
            self._loop.quit()
            return False
        elif msg.type == Gst.MessageType.EOS:
            print("End-Of-Stream reached.")
            self._loop.quit()
            return False
        elif msg.type == Gst.MessageType.LATENCY:
            try:
                self.pipeline.recalculate_latency()
            except Exception as e:
                print(f"Failed to recalculate latency: {e}")
        return True

    def dump_latency(self) -> None:
        """Dump the current pipeline latency."""
        query = Gst.Query.new_latency()
        self.pipeline.query(query)
        print(f"Pipeline latency {query.parse_latency()}")

    def play(self) -> None:
        """Start the pipeline."""
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Error starting IPC playback.")
            exit(-1)
        print("IPC consumer playing ... (ctrl+c to quit)")
        GLib.timeout_add_seconds(5, self.dump_latency)

    def stop(self) -> None:
        """Stop the pipeline."""
        print("stopping IPC consumer")
        self.pipeline.send_event(Gst.Event.new_eos())
        self.pipeline.set_state(Gst.State.NULL)
        self._loop.quit()

    def get_bus(self) -> Gst.Bus:
        """Get the GStreamer bus for the pipeline."""
        return self.pipeline.get_bus()

    def __del__(self) -> None:
        """Destructor to clean up GStreamer resources."""
        self._loop.quit()


class GstConsumer:
    """Gstreamer webrtc consumer class."""

    def __init__(
        self,
        signalling_host: str,
        signalling_port: int,
        peer_name: str,
    ) -> None:
        """Initialize the consumer with signalling server details and peer name."""
        Gst.init([])
        self._loop = GLib.MainLoop()
        self._thread_bus_calls = Thread(target=lambda: self._loop.run(), daemon=True)
        self._thread_bus_calls.start()

        self._data_channel = None

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

        # Connect early to catch webrtcbin creation and set up data channel handler
        self.source.connect("deep-element-added", self._on_element_added)

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

    def _on_element_added(
        self, _bin: Gst.Bin, _sub_bin: Gst.Bin, element: Gst.Element
    ) -> None:
        """Handle element addition to catch webrtcbin early for data channel setup."""
        if element.get_name().startswith("webrtcbin"):
            print(f"webrtcbin detected: {element.get_name()}")
            # Connect to data channel signal early
            element.connect("on-data-channel", self._on_data_channel)

    def _configure_webrtcbin(self, webrtcsrc: Gst.Element) -> None:
        if isinstance(webrtcsrc, Gst.Bin):
            webrtcbin_name = "webrtcbin0"
            webrtcbin = webrtcsrc.get_by_name(webrtcbin_name)
            assert webrtcbin is not None
            # jitterbuffer has a default 200 ms buffer.
            webrtcbin.set_property("latency", 50)

    def _on_data_channel(self, _webrtcbin: Gst.Element, channel: Gst.Element) -> None:
        """Handle incoming data channel from the server."""
        print(f"Data channel received: {channel.get_property('label')}")
        self._data_channel = channel
        channel.connect("on-open", self._on_data_channel_open)
        channel.connect("on-close", self._on_data_channel_close)
        channel.connect("on-message-string", self._on_data_channel_message)
        channel.connect("on-error", self._on_data_channel_error)

    def _on_data_channel_open(self, _channel: Gst.Element) -> None:
        """Handle data channel open event."""
        print("Data channel opened")

    def _on_data_channel_close(self, _channel: Gst.Element) -> None:
        """Handle data channel close event."""
        print("Data channel closed")
        self._data_channel = None

    def _on_data_channel_message(self, _channel: Gst.Element, message: str) -> None:
        """Handle incoming data channel message."""
        print(f"Data channel message: {message}")

    def _on_data_channel_error(self, _channel: Gst.Element, error: str) -> None:
        """Handle data channel error."""
        print(f"Data channel error: {error}")

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
    msg = bus.timed_pop_filtered(
        10 * Gst.MSECOND,
        Gst.MessageType.ERROR | Gst.MessageType.EOS | Gst.MessageType.LATENCY,
    )
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


def command_sender_loop(consumer: GstConsumer) -> None:
    """Loop to send commands over the data channel."""
    import json

    import numpy as np

    from reachy_mini.utils import create_head_pose

    freq = 0.25
    amp = 20.0

    while True:
        if consumer._data_channel:
            yaw = amp * np.sin(2.0 * np.pi * freq * time.time())
            cmd = create_head_pose(yaw=yaw, degrees=True)
            msg = {
                "set_target": cmd.tolist(),
            }
            consumer._data_channel.emit("send-string", json.dumps(msg))

        time.sleep(0.01)


def run_ipc_client() -> None:
    """Run the IPC consumer — display video from the daemon's unix-socket endpoint."""
    consumer = IpcConsumer()
    consumer.play()

    bus = consumer.get_bus()
    try:
        while True:
            if not process_msg(bus, consumer.pipeline):
                break
    except KeyboardInterrupt:
        print("User exit")
    finally:
        consumer.stop()


def run_webrtc_client(signaling_host: str, signaling_port: int) -> None:
    """Connect via WebRTC and display video/audio."""
    consumer = GstConsumer(
        signaling_host,
        signaling_port,
        "reachymini",
    )
    consumer.play()

    t = Thread(target=lambda: command_sender_loop(consumer), daemon=True)
    t.start()

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


def main() -> None:
    """Run the main function."""
    parser = argparse.ArgumentParser(
        description="GStreamer media client — test WebRTC or IPC path"
    )
    parser.add_argument(
        "--mode",
        choices=["webrtc", "ipc"],
        default="webrtc",
        help="Client mode: 'webrtc' for remote streaming, 'ipc' for local unix-socket frames (default: webrtc)",
    )
    parser.add_argument(
        "--signaling-host",
        default="127.0.0.1",
        help="Gstreamer signaling host - Reachy Mini ip (webrtc mode only)",
    )
    parser.add_argument(
        "--signaling-port",
        default=8443,
        type=int,
        help="Gstreamer signaling port (webrtc mode only)",
    )
    parser.add_argument(
        "--start-server",
        action="store_true",
        help="Start a local GstMediaServer before connecting. "
        "Useful on Lite without a running daemon.",
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Use simulated video source (requires --start-server).",
    )

    args = parser.parse_args()

    server = None
    if args.start_server:
        from reachy_mini.media.media_server import GstMediaServer

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        print("Starting local GstMediaServer...")
        server = GstMediaServer(log_level="DEBUG", use_sim=args.sim)
        server.start()
        # Give the server time to set up the IPC socket / register with signalling
        time.sleep(2)
        print(f"GstMediaServer started (camera: {server.camera_specs.name}).")

    try:
        if args.mode == "ipc":
            run_ipc_client()
        else:
            run_webrtc_client(args.signaling_host, args.signaling_port)
    finally:
        if server is not None:
            print("Stopping GstMediaServer...")
            server.stop()
            server.__del__()


if __name__ == "__main__":
    main()
