"""GStreamer UDP Frame Sender.

This module provides a class to send frames over UDP using GStreamer.
"""

import logging
from threading import Thread
from typing import Optional

import numpy as np
import numpy.typing as npt

try:
    import gi
except ImportError as e:
    raise ImportError(
        "The 'gi' module is required for GStreamerAudio but could not be imported. \
        Please check the gstreamer installation."
    ) from e

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")

from gi.repository import GLib, Gst, GstApp  # noqa: E402


class GStreamerUDPCamera:
    """A class to send frames over UDP using GStreamer."""

    def __init__(
        self,
        dest_ip: str = "127.0.0.1",
        dest_port: int = 5005,
        width: int = 1280,
        height: int = 720,
        log_level: str = "INFO",
    ) -> None:
        """Initialize the GStreamer UDP frame sender.

        Args:
            dest_ip (str): Destination IP address.
            dest_port (int): Destination UDP port.
            width (int): Width of the video frames.
            height (int): Height of the video frames.
            log_level (str): Logging level. Default: "INFO".

        """
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)

        Gst.init([])

        self.width = width
        self.height = height
        self.dest_ip = dest_ip
        self.dest_port = dest_port

        # Create GLib main loop for event handling
        self._loop = GLib.MainLoop()
        self._thread_bus_calls: Optional[Thread] = None

        # Create pipeline
        self.pipeline = Gst.Pipeline.new("udp_sender")
        self._bus = self.pipeline.get_bus()
        self._bus.add_watch(GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop)

        # Configure pipeline elements
        self._configure_pipeline()

    def _configure_pipeline(self) -> None:
        """Configure the GStreamer pipeline with individual elements."""
        self._logger.debug("Configuring UDP sender pipeline")

        # Create elements
        appsrc: GstApp = Gst.ElementFactory.make("appsrc", "src")
        capsfilter_input = Gst.ElementFactory.make("capsfilter", "capsfilter_input")
        queue = Gst.ElementFactory.make("queue", "queue")
        rtpvrawpay = Gst.ElementFactory.make("rtpvrawpay", "rtpvrawpay")
        capsfilter_rtp = Gst.ElementFactory.make("capsfilter", "capsfilter_rtp")
        udpsink = Gst.ElementFactory.make("udpsink", "udpsink")

        if not all(
            [appsrc, capsfilter_input, queue, rtpvrawpay, capsfilter_rtp, udpsink]
        ):
            raise RuntimeError("Failed to create GStreamer elements")

        # Configure appsrc
        appsrc.set_property("emit-signals", False)
        appsrc.set_property("is-live", True)
        appsrc.set_property("do-timestamp", True)
        appsrc.set_property("format", Gst.Format.TIME)

        # Configure input caps
        caps_input = Gst.Caps.from_string(
            f"video/x-raw,format=RGB,width={self.width},height={self.height},framerate=25/1"
        )
        capsfilter_input.set_property("caps", caps_input)

        # Configure queue
        queue.set_property("max-size-buffers", 2)
        queue.set_property("leaky", 2)  # downstream leak (drop old buffers)

        # Configure rtpvrawpay
        rtpvrawpay.set_property("mtu", 1400)

        # Configure RTP caps
        caps_rtp = Gst.Caps.from_string("application/x-rtp,payload=96")
        capsfilter_rtp.set_property("caps", caps_rtp)

        # Configure udpsink
        udpsink.set_property("host", self.dest_ip)
        udpsink.set_property("port", self.dest_port)
        udpsink.set_property("sync", False)

        # Add elements to pipeline
        self.pipeline.add(appsrc)
        self.pipeline.add(capsfilter_input)
        self.pipeline.add(queue)
        self.pipeline.add(rtpvrawpay)
        self.pipeline.add(capsfilter_rtp)
        self.pipeline.add(udpsink)

        # Link elements
        if not appsrc.link(capsfilter_input):
            raise RuntimeError("Failed to link appsrc to capsfilter_input")
        if not capsfilter_input.link(queue):
            raise RuntimeError("Failed to link capsfilter_input to queue")
        if not queue.link(rtpvrawpay):
            raise RuntimeError("Failed to link queue to rtpvrawpay")
        if not rtpvrawpay.link(capsfilter_rtp):
            raise RuntimeError("Failed to link rtpvrawpay to capsfilter_rtp")
        if not capsfilter_rtp.link(udpsink):
            raise RuntimeError("Failed to link capsfilter_rtp to udpsink")

        # Store appsrc reference for sending frames
        self.appsrc = appsrc

        self._logger.debug("UDP sender pipeline configured successfully")

    def _on_bus_message(
        self, bus: Gst.Bus, msg: Gst.Message, loop: GLib.MainLoop
    ) -> bool:
        """Handle GStreamer bus messages.

        Args:
            bus: GStreamer bus.
            msg: GStreamer message.
            loop: GLib main loop.

        Returns:
            bool: True to continue receiving messages, False to stop.

        """
        t = msg.type
        if t == Gst.MessageType.EOS:
            self._logger.warning("End-of-stream")
            return False
        elif t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            self._logger.error(f"Error: {err} {debug}")
            return False
        elif t == Gst.MessageType.WARNING:
            err, debug = msg.parse_warning()
            self._logger.warning(f"Warning: {err} {debug}")

        return True

    def _handle_bus_calls(self) -> None:
        """Run the GLib main loop for handling bus messages."""
        self._logger.debug("Starting bus message loop")
        self._loop.run()
        self._logger.debug("Bus message loop stopped")

    def start(self) -> None:
        """Start the UDP sender pipeline."""
        self._logger.debug("Starting UDP sender")

        # Start bus message thread
        self._thread_bus_calls = Thread(target=self._handle_bus_calls, daemon=True)
        self._thread_bus_calls.start()

        # Set pipeline to playing
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to start UDP sender pipeline")

        self._logger.info("UDP sender started")

    def send_frame(self, frame: npt.NDArray[np.uint8]) -> None:
        """Send a frame through the GStreamer pipeline.

        Args:
            frame (np.ndarray): The frame to be sent, in RGB format with shape (height, width, 3).

        Raises:
            ValueError: If frame shape doesn't match expected dimensions.

        """
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Frame shape {frame.shape} does not match expected shape "
                f"({self.height}, {self.width}, 3)"
            )

        buf = Gst.Buffer.new_wrapped(frame.tobytes())

        ret = self.appsrc.push_buffer(buf)
        if ret != Gst.FlowReturn.OK:
            self._logger.warning(f"Failed to push buffer: {ret}")

    def close(self) -> None:
        """Close the pipeline and clean up resources."""
        self._logger.debug("Closing UDP sender")

        if self.pipeline:
            # Send EOS
            self.appsrc.emit("end-of-stream")

            # Stop pipeline
            self.pipeline.set_state(Gst.State.NULL)

            # Stop main loop
            if self._loop:
                self._loop.quit()

            # Clean up bus watch
            if self._bus:
                self._bus.remove_watch()

        self._logger.info("UDP sender closed")

    def __del__(self) -> None:
        """Destructor to ensure resources are released."""
        try:
            self.close()
        except Exception:
            pass
