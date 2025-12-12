"""GStreamer camera backend.

This module provides an implementation of the CameraBase class using GStreamer.
By default the module directly returns JPEG images as output by the camera.
"""

import os
from threading import Thread
from typing import Optional, Tuple, cast

import numpy as np
import numpy.typing as npt

from reachy_mini.media.camera_constants import (
    ArducamSpecs,
    CameraResolution,
    CameraSpecs,
    ReachyMiniLiteCamSpecs,
    ReachyMiniWirelessCamSpecs,
)

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
    ) -> None:
        """Initialize the GStreamer camera."""
        super().__init__(log_level=log_level)
        Gst.init(None)
        self._loop = GLib.MainLoop()
        self._thread_bus_calls: Optional[Thread] = None

        self.pipeline = Gst.Pipeline.new("camera_recorder")

        cam_path, self.camera_specs = self.get_video_device()

        if self.camera_specs is None:
            raise RuntimeError("Camera specs not set")
        self._resolution = self.camera_specs.default_resolution
        self.resized_K = self.camera_specs.K

        if self._resolution is None:
            raise RuntimeError("Failed to get default camera resolution.")

        # note for some applications the jpeg image could be directly used
        self._appsink_video: GstApp = Gst.ElementFactory.make("appsink")
        self.set_resolution(self._resolution)
        self._appsink_video.set_property("drop", True)  # avoid overflow
        self._appsink_video.set_property("max-buffers", 1)  # keep last image only
        self.pipeline.add(self._appsink_video)

        if cam_path == "":
            self.logger.warning("Recording pipeline set without camera.")
            self.pipeline.remove(self._appsink_video)
        elif cam_path == "/tmp/reachymini_camera_socket":
            camsrc = Gst.ElementFactory.make("unixfdsrc")
            camsrc.set_property("socket-path", "/tmp/reachymini_camera_socket")
            self.pipeline.add(camsrc)
            queue = Gst.ElementFactory.make("queue")
            self.pipeline.add(queue)
            videoconvert = Gst.ElementFactory.make("videoconvert")
            self.pipeline.add(videoconvert)
            camsrc.link(queue)
            queue.link(videoconvert)
            videoconvert.link(self._appsink_video)
        elif cam_path == "imx708":
            camsrc = Gst.ElementFactory.make("libcamerasrc")
            self.pipeline.add(camsrc)
            queue = Gst.ElementFactory.make("queue")
            self.pipeline.add(queue)
            videoconvert = Gst.ElementFactory.make("videoconvert")
            self.pipeline.add(videoconvert)
            camsrc.link(queue)
            queue.link(videoconvert)
            videoconvert.link(self._appsink_video)
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

    def set_resolution(self, resolution: CameraResolution) -> None:
        """Set the camera resolution."""
        super().set_resolution(resolution)

        # Check if pipeline is not playing before changing resolution
        if self.pipeline.get_state(0).state == Gst.State.PLAYING:
            raise RuntimeError(
                "Cannot change resolution while the camera is streaming. Please close the camera first."
            )

        self._resolution = resolution
        caps_video = Gst.Caps.from_string(
            f"video/x-raw,format=BGR, width={self._resolution.value[0]},height={self._resolution.value[1]},framerate={self.framerate}/1"
        )
        self._appsink_video.set_property("caps", caps_video)

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

    def get_video_device(self) -> Tuple[str, Optional[CameraSpecs]]:
        """Use Gst.DeviceMonitor to find the unix camera path /dev/videoX.

        Returns the device path (e.g., '/dev/video2'), or '' if not found.
        """
        if os.path.exists("/tmp/reachymini_camera_socket"):
            camera_specs = cast(CameraSpecs, ReachyMiniWirelessCamSpecs)
            self.logger.debug(
                "Found wireless camera socket at /tmp/reachymini_camera_socket"
            )
            return "/tmp/reachymini_camera_socket", camera_specs

        monitor = Gst.DeviceMonitor()
        monitor.add_filter("Video/Source")
        monitor.start()

        cam_names = ["Reachy", "Arducam_12MP", "imx708"]

        devices = monitor.get_devices()
        for cam_name in cam_names:
            for device in devices:
                name = device.get_display_name()
                device_props = device.get_properties()

                if cam_name in name:
                    if device_props and device_props.has_field("api.v4l2.path"):
                        device_path = device_props.get_string("api.v4l2.path")
                        camera_specs = (
                            cast(CameraSpecs, ArducamSpecs)
                            if cam_name == "Arducam_12MP"
                            else cast(CameraSpecs, ReachyMiniLiteCamSpecs)
                        )
                        self.logger.debug(f"Found {cam_name} camera at {device_path}")
                        monitor.stop()
                        return str(device_path), camera_specs
                    elif cam_name == "imx708":
                        camera_specs = cast(CameraSpecs, ReachyMiniWirelessCamSpecs)
                        self.logger.debug(f"Found {cam_name} camera")
                        monitor.stop()
                        return cam_name, camera_specs
        monitor.stop()
        self.logger.warning("No camera found.")
        return "", None
