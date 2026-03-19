"""Shared test fixtures for unit tests."""

import os
import platform
import time
from typing import Generator, cast

import pytest

try:
    # Hack: import placo before reachy mini to fix an error with the Ubuntu CI
    import placo
except ImportError:
    pass

from reachy_mini.daemon.utils import (
    CAMERA_PIPE_NAME,
    CAMERA_SOCKET_PATH,
    is_local_camera_available,
)
from reachy_mini.media.camera_constants import (
    CameraResolution,
    CameraSpecs,
    ReachyMiniLiteCamSpecs,
)


def _gst_ipc_available() -> bool:
    """Check if GStreamer and the platform IPC sink plugin are available.

    - Linux/macOS: ``unixfdsink``
    - Windows: ``win32ipcvideosink``
    """
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init([])
        if platform.system() == "Windows":
            return Gst.ElementFactory.make("win32ipcvideosink") is not None
        else:
            return Gst.ElementFactory.make("unixfdsink") is not None
    except Exception:
        return False


@pytest.fixture()
def ipc_video_source(
    request: pytest.FixtureRequest,
) -> Generator[CameraSpecs, None, None]:
    """Start a lightweight GStreamer pipeline that pushes test frames into
    the IPC endpoint, simulating the daemon's camera branch.

    The pipeline is::

        Linux/macOS: videotestsrc → videoconvert → capsfilter(BGR) → unixfdsink
        Windows:     videotestsrc → videoconvert → capsfilter(BGR) → win32ipcvideosink

    Use the ``ipc_resolution`` marker to override the default resolution::

        @pytest.mark.ipc_resolution(CameraResolution.R1280x720at30fps)
        def test_something(ipc_video_source): ...

    Yields the ``CameraSpecs`` used so tests can pass them to
    ``GStreamerCamera`` or ``MediaManager``.
    """
    if not _gst_ipc_available():
        pytest.skip("GStreamer or IPC sink plugin not available")

    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init([])

    specs: CameraSpecs = cast(CameraSpecs, ReachyMiniLiteCamSpecs)

    # Allow per-test resolution override via marker
    marker = request.node.get_closest_marker("ipc_resolution")
    if marker and marker.args:
        resolution: CameraResolution = marker.args[0]
    else:
        resolution = specs.default_resolution

    w, h, fps = resolution.value[0], resolution.value[1], int(resolution.value[2])

    is_windows = platform.system() == "Windows"

    if is_windows:
        ipc_path = CAMERA_PIPE_NAME
    else:
        ipc_path = CAMERA_SOCKET_PATH
        # Clean up stale socket
        if os.path.exists(CAMERA_SOCKET_PATH):
            os.remove(CAMERA_SOCKET_PATH)

    # Build the pipeline programmatically instead of with parse_launch
    # because GStreamer's pipeline description parser interprets backslashes
    # in the Windows named-pipe path (\\.\pipe\...) as escape characters.
    pipeline = Gst.Pipeline.new("ipc_fixture")

    src = Gst.ElementFactory.make("videotestsrc")
    src.set_property("is-live", True)
    src.set_property("pattern", "smpte")

    convert = Gst.ElementFactory.make("videoconvert")

    caps = Gst.Caps.from_string(
        f"video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1"
    )
    capsfilter = Gst.ElementFactory.make("capsfilter")
    capsfilter.set_property("caps", caps)

    if is_windows:
        sink = Gst.ElementFactory.make("win32ipcvideosink")
        sink.set_property("pipe-name", CAMERA_PIPE_NAME)
    else:
        sink = Gst.ElementFactory.make("unixfdsink")
        sink.set_property("socket-path", CAMERA_SOCKET_PATH)

    for elem in [src, convert, capsfilter, sink]:
        pipeline.add(elem)
    src.link(convert)
    convert.link(capsfilter)
    capsfilter.link(sink)

    pipeline.set_state(Gst.State.PLAYING)

    # Wait for the IPC endpoint to become available.
    # On Windows, os.path.exists() is unreliable for named pipes, so we
    # use is_local_camera_available() which uses CreateFileW internally.
    for _ in range(40):
        if is_local_camera_available():
            break
        time.sleep(0.1)
    else:
        pipeline.set_state(Gst.State.NULL)
        pytest.fail(f"IPC endpoint {ipc_path} was not created in time")

    # Give a moment for the first buffers to flow
    time.sleep(0.5)

    yield specs

    pipeline.set_state(Gst.State.NULL)
    if not is_windows and os.path.exists(CAMERA_SOCKET_PATH):
        os.remove(CAMERA_SOCKET_PATH)
