"""GStreamer local camera backend (IPC reader).

Reads camera frames from the local IPC endpoint exposed by the WebRTC
daemon:

* **Linux / macOS**: Unix domain socket via ``unixfdsrc``
* **Windows**: Win32 named shared memory via ``win32ipcvideosrc``

Frames arrive in BGR format from the daemon's IPC branch, so no
``videoconvert`` is needed on the reader side — the appsink receives
BGR data directly and converts it to a NumPy array.

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

import logging
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
    MujocoCameraSpecs,
    ReachyMiniWirelessCamSpecs,
)
from reachy_mini.media.camera_utils import scale_intrinsics

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


class GStreamerCamera:
    """Camera that reads BGR frames from the daemon's local IPC endpoint.

    The WebRTC daemon exposes BGR camera frames via a local IPC mechanism:

    * Linux / macOS: ``unixfdsink`` / ``unixfdsrc`` (Unix domain socket)
    * Windows: ``win32ipcvideosink`` / ``win32ipcvideosrc`` (shared memory)

    Since the daemon's IPC branch already converts to BGR, the reader
    pipeline is simply ``source → queue → appsink`` with no extra
    conversion.

    Attributes:
        camera_specs: Camera specifications (resolutions, intrinsics, …).
        resized_K: Intrinsic matrix rescaled to the current resolution.

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
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        Gst.init([])
        self._loop = GLib.MainLoop()
        self._thread_bus_calls: Optional[Thread] = None

        # Default to wireless specs (daemon manages the camera)
        self.camera_specs: CameraSpecs = cast(CameraSpecs, ReachyMiniWirelessCamSpecs)
        self._resolution: Optional[CameraResolution] = (
            self.camera_specs.default_resolution
        )
        self.resized_K: Optional[npt.NDArray[np.float64]] = self.camera_specs.K

        self.pipeline = Gst.Pipeline.new("camera_ipc_reader")

        # Create appsink for frame output
        self._appsink_video: GstApp = Gst.ElementFactory.make("appsink")
        self.set_resolution(self._resolution)
        self._appsink_video.set_property("drop", True)
        self._appsink_video.set_property("max-buffers", 1)
        self.pipeline.add(self._appsink_video)

        # Build platform-specific IPC source pipeline
        self._build_ipc_source()

    # ------------------------------------------------------------------
    # Resolution / calibration properties
    # ------------------------------------------------------------------

    @property
    def resolution(self) -> tuple[int, int]:
        """Current resolution as ``(width, height)`` in pixels.

        Raises:
            RuntimeError: If the resolution has not been set yet.

        """
        if self._resolution is None:
            raise RuntimeError("Camera resolution is not set.")
        return (self._resolution.value[0], self._resolution.value[1])

    @property
    def framerate(self) -> int:
        """Current frame rate in fps.

        Raises:
            RuntimeError: If the resolution has not been set yet.

        """
        if self._resolution is None:
            raise RuntimeError("Camera resolution is not set.")
        return int(self._resolution.value[2])

    @property
    def K(self) -> Optional[npt.NDArray[np.float64]]:
        """Camera intrinsic matrix rescaled to the current resolution.

        Returns the 3×3 matrix::

            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]

        or ``None`` if not yet available.

        """
        return self.resized_K

    @property
    def D(self) -> Optional[npt.NDArray[np.float64]]:
        """Distortion coefficients ``[k1, k2, p1, p2, k3]``, or ``None``."""
        if self.camera_specs is not None:
            return self.camera_specs.D
        return None

    def set_resolution(self, resolution: CameraResolution) -> None:
        """Change the camera resolution.

        Updates the appsink caps and rescales the intrinsic matrix.
        If the pipeline is already playing it is restarted automatically.

        Args:
            resolution: Desired resolution from ``CameraResolution``.

        Raises:
            RuntimeError: If camera specs are not set or if the camera is a
                MuJoCo simulated camera (resolution change not supported).
            ValueError: If the resolution is not in the camera's
                ``available_resolutions``.

        """
        if self.camera_specs is None:
            raise RuntimeError(
                "Camera specs not set. Open the camera before setting the resolution."
            )

        if isinstance(self.camera_specs, MujocoCameraSpecs):
            raise RuntimeError(
                "Cannot change resolution of Mujoco simulated camera for now."
            )

        if resolution not in self.camera_specs.available_resolutions:
            raise ValueError(
                f"Resolution not supported by the camera. "
                f"Available resolutions are: {self.camera_specs.available_resolutions}"
            )

        # Rescale intrinsic matrix
        original_K = self.camera_specs.K
        original_size: tuple[int, int] = (
            CameraResolution.R3840x2592at30fps.value[0],
            CameraResolution.R3840x2592at30fps.value[1],
        )
        target_size: tuple[int, int] = (resolution.value[0], resolution.value[1])
        crop_scale = resolution.value[3]
        self.resized_K = scale_intrinsics(
            original_K, original_size, target_size, crop_scale
        )

        # Restart pipeline if needed
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

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_ipc_source(self) -> None:
        """Build the IPC source pipeline for the current platform.

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

    # ------------------------------------------------------------------
    # Bus handling
    # ------------------------------------------------------------------

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

    def _dump_latency(self) -> None:
        query = Gst.Query.new_latency()
        self.pipeline.query(query)
        self.logger.info(f"Pipeline latency {query.parse_latency()}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Start the GStreamer pipeline and begin receiving frames."""
        self.pipeline.set_state(Gst.State.PLAYING)
        self._thread_bus_calls = Thread(target=self._handle_bus_calls, daemon=True)
        self._thread_bus_calls.start()
        GLib.timeout_add_seconds(5, self._dump_latency)

    def read(self) -> Optional[npt.NDArray[np.uint8]]:
        """Pull the latest BGR frame from the IPC endpoint.

        Returns:
            A NumPy array of shape ``(height, width, 3)`` in BGR order,
            or ``None`` if no frame is available within the timeout.

        """
        sample = self._appsink_video.try_pull_sample(20_000_000)
        if sample is None:
            return None

        buf = sample.get_buffer()
        if buf is None:
            self.logger.warning("Buffer is None")
            return None
        data = buf.extract_dup(0, buf.get_size())

        arr = np.frombuffer(data, dtype=np.uint8).reshape(
            (self.resolution[1], self.resolution[0], 3)
        )
        return arr

    def close(self) -> None:
        """Stop the pipeline and release resources."""
        self._loop.quit()
        self.pipeline.set_state(Gst.State.NULL)
