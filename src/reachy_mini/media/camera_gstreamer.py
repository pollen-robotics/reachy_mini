"""GStreamer local camera backend (IPC reader).

This module provides an implementation of the CameraBase class that reads
camera frames from the local IPC endpoint exposed by the WebRTC daemon:
- Linux/macOS: Unix domain socket (unixfdsrc)
- Windows: Win32 named shared memory (win32ipcvideosrc)

Frames arrive in BGR format from the daemon's IPC branch, so no
``videoconvert`` is needed on the reader side — the appsink receives
BGR data directly and converts it to a NumPy array.

This backend is used by the LOCAL media backend when the SDK client runs
on the same machine as the daemon. It avoids the overhead of WebRTC
encoding/decoding for on-device applications.

Example usage:
    >>> from reachy_mini.media.camera_gstreamer import GStreamerCamera
    >>>
    >>> camera = GStreamerCamera(log_level="INFO")
    >>> camera.open()
    >>> frame = camera.read()
    >>> if frame is not None:
    ...     print(f"Captured frame with shape: {frame.shape}")
    >>> camera.close()
"""

import platform
from threading import Thread
from typing import Optional, cast

import numpy as np
import numpy.typing as npt

from reachy_mini.daemon.utils import (
    CAMERA_PIPE_NAME,
    CAMERA_SOCKET_PATH,
)
from reachy_mini.media.camera_constants import (
    CameraResolution,
    CameraSpecs,
    ReachyMiniWirelessCamSpecs,
)

try:
    import gi
except ImportError as e:
    raise ImportError(
        "The 'gi' module is required for GStreamerCamera but could not be imported. "
        "Please check the gstreamer installation."
    ) from e

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")

from gi.repository import GLib, Gst, GstApp  # noqa: E402

from .camera_base import CameraBase  # noqa: E402


class GStreamerCamera(CameraBase):
    """Camera implementation that reads from the daemon's local IPC endpoint.

    The WebRTC daemon exposes BGR camera frames via a local IPC mechanism:
    - Linux/macOS: unixfdsink/unixfdsrc (Unix domain socket)
    - Windows: win32ipcvideosink/win32ipcvideosrc (shared memory)

    Since the daemon's IPC branch already converts to BGR, the reader
    pipeline is simply ``source → appsink`` with no extra conversion.
    """

    def __init__(
        self,
        log_level: str = "INFO",
    ) -> None:
        """Initialize the GStreamer local camera reader.

        Args:
            log_level: Logging level for camera operations.

        Raises:
            RuntimeError: If the IPC source element cannot be created.

        """
        super().__init__(log_level=log_level)
        Gst.init([])
        self._loop = GLib.MainLoop()
        self._thread_bus_calls: Optional[Thread] = None

        # Default to wireless specs (daemon manages the camera)
        self.camera_specs: CameraSpecs = cast(CameraSpecs, ReachyMiniWirelessCamSpecs)
        self._resolution = self.camera_specs.default_resolution
        self.resized_K = self.camera_specs.K

        self.pipeline = Gst.Pipeline.new("camera_ipc_reader")

        # Create appsink for frame output
        self._appsink_video: GstApp = Gst.ElementFactory.make("appsink")
        self.set_resolution(self._resolution)
        self._appsink_video.set_property("drop", True)
        self._appsink_video.set_property("max-buffers", 1)
        self.pipeline.add(self._appsink_video)

        # Build platform-specific IPC source pipeline
        self._build_ipc_source()

    def _build_ipc_source(self) -> None:
        """Build the IPC source pipeline based on the current platform.

        The daemon's IPC branch already outputs BGR frames, so no
        ``videoconvert`` is needed here — just ``source → queue → appsink``.
        """
        if platform.system() == "Windows":
            camsrc = Gst.ElementFactory.make("win32ipcvideosrc")
            if camsrc is None:
                raise RuntimeError(
                    "Failed to create win32ipcvideosrc. "
                    "Is the win32ipc GStreamer plugin installed?"
                )
            camsrc.set_property("pipe-name", CAMERA_PIPE_NAME)
        else:
            # Linux and macOS use unixfdsrc
            camsrc = Gst.ElementFactory.make("unixfdsrc")
            if camsrc is None:
                raise RuntimeError(
                    "Failed to create unixfdsrc. "
                    "Is the unixfd GStreamer plugin installed?"
                )
            camsrc.set_property("socket-path", CAMERA_SOCKET_PATH)

        queue = Gst.ElementFactory.make("queue")
        if queue is None:
            raise RuntimeError("Failed to create GStreamer queue element")

        self.pipeline.add(camsrc)
        self.pipeline.add(queue)

        camsrc.link(queue)
        queue.link(self._appsink_video)

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

        should_restart = False
        if self.pipeline.get_state(0).state == Gst.State.PLAYING:
            self.close()
            should_restart = True

        self._resolution = resolution
        caps_video = Gst.Caps.from_string(
            f"video/x-raw,format=BGR,"
            f"width={self._resolution.value[0]},"
            f"height={self._resolution.value[1]},"
            f"framerate={self.framerate}/1"
        )
        self._appsink_video.set_property("caps", caps_video)

        if should_restart:
            self.open()

    def _dump_latency(self) -> None:
        query = Gst.Query.new_latency()
        self.pipeline.query(query)
        self.logger.info(f"Pipeline latency {query.parse_latency()}")

    def open(self) -> None:
        """Open the camera by starting the GStreamer pipeline."""
        self.pipeline.set_state(Gst.State.PLAYING)
        self._thread_bus_calls = Thread(target=self._handle_bus_calls, daemon=True)
        self._thread_bus_calls.start()
        GLib.timeout_add_seconds(5, self._dump_latency)

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
        """Read a frame from the camera via IPC.

        Returns:
            The captured BGR frame as a NumPy array, or None if no frame is available.

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
