"""GStreamer local camera backend (IPC reader).

Reads camera frames from the local IPC endpoint exposed by the WebRTC
daemon:

* **Linux / macOS**: Unix domain socket via ``unixfdsrc``
* **Windows**: Win32 named shared memory via ``win32ipcvideosrc``

A ``v4l2convert`` (hardware-accelerated on RPi) or ``videoconvert``
(software fallback) element converts the daemon's native frame format
to BGR before the appsink, so the reader works regardless of what
format the daemon sends (YUY2 from RPi libcamerasrc, BGR from Lite
v4l2src, I420 from simulation, etc.).  When the daemon already sends
BGR the converter runs in passthrough mode (zero-copy).

This backend is used by the ``LOCAL`` media backend when the SDK client
runs on the same machine as the daemon.  It avoids the overhead of WebRTC
encoding / decoding for on-device applications.

Resolution management
~~~~~~~~~~~~~~~~~~~~~
The camera intrinsic matrix ``K`` is automatically rescaled whenever the
resolution changes (see ``set_resolution``).  Distortion coefficients
``D`` come directly from the ``CameraSpecs`` dataclass and are
resolution-independent.

The ``MujocoCameraSpecs`` camera does not support runtime resolution
changes — ``set_resolution`` will raise ``RuntimeError`` for it.

Example usage::

    from reachy_mini.media.camera_gstreamer import GStreamerCamera

    camera = GStreamerCamera(log_level="INFO")
    camera.open()
    frame = camera.read()
    if frame is not None:
        print(f"Captured frame with shape: {frame.shape}")
    camera.close()
"""

import platform
import time
from threading import Thread
from typing import Optional, cast

import numpy as np
import numpy.typing as npt

from reachy_mini.daemon.utils import (
    CAMERA_PIPE_NAME,
    CAMERA_SOCKET_PATH,
)
from reachy_mini.media.camera_base import CameraBase
from reachy_mini.media.camera_constants import (
    CameraResolution,
    CameraSpecs,
    ReachyMiniLiteCamSpecs,
)
from reachy_mini.media.gstreamer_utils import get_sample

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


class GStreamerCamera(CameraBase):
    """Camera that reads BGR frames from the daemon's local IPC endpoint.

    The WebRTC daemon exposes BGR camera frames via a local IPC mechanism:

    * Linux / macOS: ``unixfdsink`` / ``unixfdsrc`` (Unix domain socket)
    * Windows: ``win32ipcvideosink`` / ``win32ipcvideosrc`` (shared memory)

    Since the daemon's IPC branch already converts to BGR, the reader
    pipeline is simply ``source → queue → appsink`` with no extra
    conversion.

    """

    def __init__(
        self,
        log_level: str = "INFO",
        camera_specs: Optional[CameraSpecs] = None,
    ) -> None:
        """Initialize the GStreamer local camera reader.

        Args:
            log_level: Logging level for camera operations.
            camera_specs: Camera specifications detected by the daemon.
                When ``None`` falls back to ``ReachyMiniLiteCamSpecs``
                with a warning (e.g. direct instantiation without SDK).

        Raises:
            RuntimeError: If the IPC source element cannot be created.

        """
        super().__init__(log_level=log_level)

        Gst.init([])
        self._loop = GLib.MainLoop()
        self._thread_bus_calls: Optional[Thread] = None

        if camera_specs is not None:
            self.camera_specs = camera_specs
        else:
            self.logger.warning(
                "No camera_specs provided — defaulting to ReachyMiniLiteCamSpecs."
            )
            self.camera_specs = cast(CameraSpecs, ReachyMiniLiteCamSpecs)
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

    def _apply_resolution(self, resolution: CameraResolution) -> None:
        """Apply resolution: restart the pipeline if it is already playing."""
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

    def _build_ipc_source(self) -> None:
        """Build the IPC source pipeline for the current platform.

        Pipeline: ``source → queue → v4l2convert/videoconvert → appsink``

        The converter ensures BGR output regardless of the daemon's native
        format.  ``v4l2convert`` is preferred (hardware-accelerated on RPi);
        ``videoconvert`` is the software fallback.  When the daemon already
        sends BGR the converter runs in passthrough mode (zero-copy).
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

        # Prefer v4l2convert (hardware-accelerated on RPi), fall back to
        # videoconvert (software) on other platforms.
        try:
            convert = Gst.ElementFactory.make("v4l2convert")
        except Exception:
            convert = None
        if convert is None:
            self.logger.debug(
                "v4l2convert not available, falling back to videoconvert."
            )
            convert = Gst.ElementFactory.make("videoconvert")
        if convert is None:
            raise RuntimeError("Failed to create video converter element")

        self.pipeline.add(camsrc)
        self.pipeline.add(queue)
        self.pipeline.add(convert)

        camsrc.link(queue)
        queue.link(convert)
        convert.link(self._appsink_video)

    def _on_bus_message(self, bus: Gst.Bus, msg: Gst.Message, loop) -> bool:  # type: ignore[no-untyped-def]
        t = msg.type
        if t == Gst.MessageType.EOS:
            self.logger.warning("End-of-stream")
            return False
        elif t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            self.logger.warning(
                f"GStreamer pipeline error (domain={err.domain}, code={err.code}): {err.message}"
            )
            self.logger.debug(f"GStreamer error debug info: {debug}")
            # Keep the bus watch active — some errors are transient and the pipeline
            # will self-recover. Fatal errors should be handled by inspecting
            # err.domain and err.code.
            return True
        return True

    def _handle_bus_calls(self) -> None:
        self.logger.debug("starting bus message loop")
        bus = self.pipeline.get_bus()
        bus.add_watch(GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop)
        self._loop.run()
        bus.remove_watch()
        self.logger.debug("bus message loop stopped")

    def _dump_latency(self) -> None:
        query = Gst.Query.new_latency()
        self.pipeline.query(query)
        self.logger.info(f"Pipeline latency {query.parse_latency()}")

    def open(self) -> None:
        """Start the GStreamer pipeline and begin receiving frames."""
        self.pipeline.set_state(Gst.State.PLAYING)
        self._thread_bus_calls = Thread(target=self._handle_bus_calls, daemon=True)
        self._thread_bus_calls.start()
        GLib.timeout_add_seconds(5, self._dump_latency)
        # Best-effort wait for the first frame before returning, so callers can
        # read immediately without getting None.
        deadline = time.monotonic() + 2.0
        try:
            while time.monotonic() < deadline:
                if self._appsink_video.emit("try-pull-sample", 100_000_000) is not None:
                    break
        except Exception:
            pass

    def read(self) -> Optional[npt.NDArray[np.uint8]]:
        """Pull the latest BGR frame from the IPC endpoint.

        Returns:
            A NumPy array of shape ``(height, width, 3)`` in BGR order,
            or ``None`` if no frame is available within the timeout.

        """
        data = get_sample(self._appsink_video, self.logger)
        if data is None:
            return None
        return np.frombuffer(data, dtype=np.uint8).reshape(
            (self.resolution[1], self.resolution[0], 3)
        )

    def close(self) -> None:
        """Stop the pipeline and release resources."""
        self._loop.quit()
        self.pipeline.set_state(Gst.State.NULL)
