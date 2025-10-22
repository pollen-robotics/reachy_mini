"""GStreamer camera backend.

This module provides an implementation of the CameraBase class using GStreamer.
By default the module directly returns JPEG images as output by the camera.
"""

from threading import Thread
from typing import Optional

import numpy as np
import numpy.typing as npt

from reachy_mini.media.camera_constants import CameraResolution

try:
    import gi
except ImportError as e:
    raise ImportError(
        "The 'gi' module is required for GStreamerCamera but could not be imported. \
                      Please install the GStreamer backend: pip install .[gstreamer]."
    ) from e

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")


from gi.repository import GLib, Gst, GstApp  # noqa: E402

from .camera_base import CameraBase  # noqa: E402


class GStreamerCamera(CameraBase):
    """Camera implementation using GStreamer."""

    def __init__(
        self,
        log_level: str = "INFO",
        resolution: CameraResolution = CameraResolution.R1280x720,
    ) -> None:
        """Initialize the GStreamer camera."""
        super().__init__(log_level=log_level, resolution=resolution)
        Gst.init(None)
        self._loop = GLib.MainLoop()
        self._thread_bus_calls: Optional[Thread] = None

        self.pipeline = Gst.Pipeline.new("camera_recorder")

        # note for some applications the jpeg image could be directly used
        self._appsink_video: GstApp = Gst.ElementFactory.make("appsink")
        caps_video = Gst.Caps.from_string(
            f"video/x-raw,format=BGR, width={self.resolution[0]},height={self.resolution[1]},framerate={self.framerate}/1"
        )
        self._appsink_video.set_property("caps", caps_video)
        self._appsink_video.set_property("drop", True)  # avoid overflow
        self._appsink_video.set_property("max-buffers", 1)  # keep last image only
        self.pipeline.add(self._appsink_video)

        cam_path = self.get_arducam_video_device()
        if cam_path == "":
            self.logger.warning("Recording pipeline set without camera.")
            self.pipeline.remove(self._appsink_video)
        else:
            camsrc = Gst.ElementFactory.make("v4l2src")
            camsrc.set_property("device", cam_path)
            self.pipeline.add(camsrc)
            queue = Gst.ElementFactory.make("queue")
            self.pipeline.add(queue)
            # use vaapijpegdec or nvjpegdec for hardware acceleration
            jpegdec = Gst.ElementFactory.make("jpegdec")
            self.pipeline.add(jpegdec)
            videoconvert = Gst.ElementFactory.make("videoconvert")
            self.pipeline.add(videoconvert)
            camsrc.link(queue)
            queue.link(jpegdec)
            jpegdec.link(videoconvert)
            videoconvert.link(self._appsink_video)

    def _on_bus_message(self, bus: Gst.Bus, msg: Gst.Message, loop) -> bool:  # type: ignore[no-untyped-def]
        t = msg.type
        if t == Gst.MessageType.EOS:
            self.logger.warning("End-of-stream")
            return False

        elif t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            self.logger.error(f"Error: {err} {debug}")
            return False

        return True

    def _handle_bus_calls(self) -> None:
        self.logger.debug("starting bus message loop")
        bus = self.pipeline.get_bus()
        bus.add_watch(GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop)
        self._loop.run()
        bus.remove_watch()
        self.logger.debug("bus message loop stopped")

    def open(self) -> None:
        """Open the camera using GStreamer."""
        self.pipeline.set_state(Gst.State.PLAYING)
        self._thread_bus_calls = Thread(target=self._handle_bus_calls, daemon=True)
        self._thread_bus_calls.start()

    def _get_sample(self, appsink: GstApp.AppSink) -> Optional[bytes]:
        sample = appsink.try_pull_sample(20_000_000)
        if sample is None:
            return None
        data = None
        if isinstance(sample, Gst.Sample):
            buf = sample.get_buffer()
            if buf is None:
                self.logger.warning("Buffer is None")

            data = buf.extract_dup(0, buf.get_size())
        return data

    def read(self) -> Optional[npt.NDArray[np.uint8]]:
        """Read a frame from the camera. Returns the frame or None if error.

        Returns:
            Optional[npt.NDArray[np.uint8]]: The captured BGR frame as a NumPy array, or None if error.

        """
        data = self._get_sample(self._appsink_video)
        if data is None:
            return None

        arr = np.frombuffer(data, dtype=np.uint8).reshape(
            (self.resolution[1], self.resolution[0], 3)
        )
        return arr

    def close(self) -> None:
        """Release the camera resource."""
        self._loop.quit()
        self.pipeline.set_state(Gst.State.NULL)

    def get_arducam_video_device(self) -> str:
        """Use Gst.DeviceMonitor to find the unix camera path /dev/videoX of the Arducam_12MP webcam.

        Returns the device path (e.g., '/dev/video2'), or '' if not found.
        """
        monitor = Gst.DeviceMonitor()
        monitor.add_filter("Video/Source")
        monitor.start()

        devices = monitor.get_devices()
        for device in devices:
            name = device.get_display_name()
            device_props = device.get_properties()
            if name and "Arducam_12MP" in name:
                if device_props and device_props.has_field("api.v4l2.path"):
                    device_path = device_props.get_string("api.v4l2.path")
                    self.logger.debug(f"Found Arducam_12MP at {device_path}")
                    monitor.stop()
                    return str(device_path)
        monitor.stop()
        self.logger.warning("Arducam_12MP webcam not found.")
        return ""
