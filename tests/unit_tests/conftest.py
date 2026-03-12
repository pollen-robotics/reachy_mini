"""Shared test fixtures for unit tests."""

import os
import platform
import time
from typing import Generator, cast

import pytest

from reachy_mini.daemon.utils import CAMERA_SOCKET_PATH
from reachy_mini.media.camera_constants import (
    CameraResolution,
    CameraSpecs,
    ReachyMiniLiteCamSpecs,
)


def _gst_available() -> bool:
    """Check if GStreamer and the unixfdsink plugin are available."""
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init([])
        sink = Gst.ElementFactory.make("unixfdsink")
        return sink is not None
    except Exception:
        return False


@pytest.fixture()
def ipc_video_source(
    request: pytest.FixtureRequest,
) -> Generator[CameraSpecs, None, None]:
    """Start a lightweight GStreamer pipeline that pushes test frames into
    the IPC socket, simulating the daemon's camera branch.

    The pipeline is::

        videotestsrc → videoconvert → capsfilter(BGR) → unixfdsink

    Use the ``ipc_resolution`` marker to override the default resolution::

        @pytest.mark.ipc_resolution(CameraResolution.R1280x720at30fps)
        def test_something(ipc_video_source): ...

    Yields the ``CameraSpecs`` used so tests can pass them to
    ``GStreamerCamera`` or ``MediaManager``.
    """
    if platform.system() == "Windows":
        pytest.skip("IPC video fixture only supports Unix (unixfdsink)")
    if not _gst_available():
        pytest.skip("GStreamer or unixfdsink plugin not available")

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

    # Clean up stale socket
    if os.path.exists(CAMERA_SOCKET_PATH):
        os.remove(CAMERA_SOCKET_PATH)

    pipeline = Gst.parse_launch(
        f"videotestsrc is-live=true pattern=smpte "
        f"! videoconvert "
        f"! video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 "
        f"! unixfdsink socket-path={CAMERA_SOCKET_PATH}"
    )
    pipeline.set_state(Gst.State.PLAYING)

    # Wait for the socket to appear and the first buffer to flow
    for _ in range(20):
        if os.path.exists(CAMERA_SOCKET_PATH):
            break
        time.sleep(0.1)
    else:
        pipeline.set_state(Gst.State.NULL)
        pytest.fail(f"IPC socket {CAMERA_SOCKET_PATH} was not created in time")

    # Give a moment for the first buffers to flow
    time.sleep(0.5)

    yield specs

    pipeline.set_state(Gst.State.NULL)
    if os.path.exists(CAMERA_SOCKET_PATH):
        os.remove(CAMERA_SOCKET_PATH)
