"""Shared test fixtures for unit tests."""

import os
import platform
import struct
import time
import types
from array import array
from typing import Generator, cast

import pytest
import usb.util

from reachy_mini.media.audio_control_utils import CONTROL_SUCCESS, ReSpeaker


def _webrtc_plugin_available() -> bool:
    """True when the gst-plugins-rs webrtc elements register (libgstrswebrtc.so)."""
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init([])
    except Exception:
        return False
    return all(
        Gst.ElementFactory.find(name) is not None
        for name in ("webrtcsrc", "webrtcsink")
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip ``webrtc`` tests when the plugin is missing — unless it was asked for.

    When the plugin isn't installed the webrtc-marked tests skip cleanly (local
    dev, macOS). But when the run explicitly selects them (``-m webrtc``), a
    missing plugin means a broken setup, not a reason to skip: we leave them
    selected so they run and fail loudly (their GstWebRTCClient/GstMediaServer
    construction and the gst-launch producer error out on the missing elements).
    """
    if _webrtc_plugin_available():
        return
    markexpr = config.getoption("markexpr") or ""
    if "webrtc" in markexpr and "not webrtc" not in markexpr:
        return  # explicitly requested -> run and fail loudly
    skip = pytest.mark.skip(
        reason="gst-plugins-rs webrtc plugin unavailable (libgstrswebrtc.so not loaded)"
    )
    for item in items:
        if item.get_closest_marker("webrtc") is not None:
            item.add_marker(skip)


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


@pytest.fixture()
def audio_loopback() -> None:
    """Gate tests that need the capture side to loop back the playback side.

    The CI/docker harness sets up a PulseAudio null sink whose ``.monitor``
    captures whatever is played (via the ``type pulse`` ``~/.asoundrc``), so a
    play -> record round-trip can assert real audio reached the sink. It signals
    that by exporting ``REACHY_TEST_AUDIO_LOOPBACK=1``. Skip otherwise — on real
    hardware the mic is not a loopback of the speaker.
    """
    if os.environ.get("REACHY_TEST_AUDIO_LOOPBACK") != "1":
        pytest.skip(
            "no virtual audio loopback (set REACHY_TEST_AUDIO_LOOPBACK=1 via the "
            "CI/docker harness)"
        )


class _FakeXVF3800:
    """In-memory stand-in for the ReSpeaker/XMOS USB device (no hardware).

    The whole board is reached through one ``ctrl_transfer`` seam, so a register
    file keyed by ``(resid, cmdid)`` reproduces the vendor protocol: an OUT
    stores the raw payload, an IN returns a success status byte + the stored
    payload (zeros if never written). Writes therefore round-trip through reads,
    which is what ``apply_audio_config``'s verify step relies on.
    """

    def __init__(self) -> None:
        self.regs: dict[tuple[int, int], bytes] = {}
        # ReSpeaker.close() -> usb.util.dispose_resources(dev) -> dev._ctx.dispose
        self._ctx = types.SimpleNamespace(dispose=lambda dev: None)

    def ctrl_transfer(
        self,
        bmRequestType,
        bRequest,
        wValue=0,
        wIndex=0,
        data_or_wLength=None,
        timeout=0,
    ):
        key = (wIndex, wValue & 0x7F)  # (resid, cmdid); read sets 0x80 on cmdid
        if bmRequestType & usb.util.CTRL_IN:
            length = int(data_or_wLength)
            payload = self.regs.get(key, b"")[: length - 1].ljust(length - 1, b"\x00")
            return array("B", bytes([CONTROL_SUCCESS]) + payload)
        self.regs[key] = bytes(data_or_wLength)  # OUT: store the written payload
        return len(self.regs[key])


@pytest.fixture()
def fake_respeaker() -> ReSpeaker:
    """A ``ReSpeaker`` backed by a fake XVF3800 — exercises the DoA/config/USB
    protocol logic with no board. DOA_VALUE_RADIANS is seeded with a known
    ``(angle, speech)`` (resid 20, cmdid 19); rw params round-trip on write.
    """
    dev = _FakeXVF3800()
    dev.regs[(20, 19)] = struct.pack("<ff", 1.57, 1.0)  # angle=1.57 rad, speech=True
    return ReSpeaker(dev)


@pytest.fixture()
def sim_backend() -> Generator[object, None, None]:
    """Yield a headless ``MockupSimBackend(use_audio=False)`` — no hardware, no audio.

    Its FK/IK use the pure-Python analytical kinematics; ``_media_server`` is a
    MagicMock (with ``camera_specs``) so the media-delegating and head-tracking
    branches are exercisable without a camera. ``current_head_pose`` is seeded so
    ``get_present_head_pose()`` doesn't assert. This is the shared vehicle for
    the ``daemon/backend/abstract.py`` tests (replaces the ad-hoc per-file
    ``_make_backend()`` helpers).
    """
    from unittest.mock import MagicMock

    import numpy as np

    from reachy_mini.daemon.backend.mockup_sim.backend import MockupSimBackend

    backend = MockupSimBackend(use_audio=False)
    backend._media_server = MagicMock()
    backend.current_head_pose = np.eye(4)
    try:
        yield backend
    finally:
        backend.close()


@pytest.fixture()
def router_app():
    """Build a TestClient with a single router mounted on a bare FastAPI app.

    Removes the need for a real daemon: pass ``backend`` / ``daemon`` stubs and
    they're wired both via ``app.dependency_overrides`` (for ``Depends`` seams)
    and on ``app.state.daemon`` (for routers that read it directly). Extra
    dependency overrides can be passed as ``overrides={dep_fn: provider}``.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from reachy_mini.daemon.app.dependencies import (
        get_backend,
        get_daemon,
        ws_get_backend,
    )

    def _make(router, *, backend=None, daemon=None, overrides=None) -> TestClient:
        app = FastAPI()
        app.include_router(router)
        if backend is not None:
            app.dependency_overrides[get_backend] = lambda: backend
            app.dependency_overrides[ws_get_backend] = lambda: backend
        if daemon is not None:
            app.dependency_overrides[get_daemon] = lambda: daemon
            app.state.daemon = daemon
        for dep, provider in (overrides or {}).items():
            app.dependency_overrides[dep] = provider
        return TestClient(app)

    return _make
