"""Unit tests for the MJPEG decode-rate cap in ``GstMediaServer``."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import cast
from unittest.mock import MagicMock

import pytest

from reachy_mini.media import media_server as ms
from reachy_mini.media.camera_constants import CameraResolution
from reachy_mini.media.media_server import GstMediaServer


def _make_server(resolution: CameraResolution) -> GstMediaServer:
    """Build a minimal ``GstMediaServer`` with only the video attrs wired up.

    Bypasses ``__init__`` (which boots GStreamer and opens devices) and sets
    just what ``framerate`` / ``decode_framerate`` / ``resolution`` /
    ``_build_v4l2_source`` touch.
    """
    server = cast(GstMediaServer, object.__new__(GstMediaServer))
    server._logger = logging.getLogger("test_decode_fps")
    server._resolution = resolution
    # Stubs for `__del__` -> `close()`, which the destructor calls at GC time.
    server._loop = MagicMock()
    server._bus_sender = MagicMock()
    return server


@pytest.mark.parametrize(
    ("resolution", "cap", "expected"),
    [
        # A 60 fps camera mode reports the capped rate.
        (CameraResolution.R1920x1080at60fps, 10, 10),
        # The cap only lowers the rate: a 10 fps mode stays at 10 fps under a 30 cap.
        (CameraResolution.R3840x2160at10fps, 30, 10),
    ],
)
def test_decode_framerate_is_min_of_camera_rate_and_cap(
    monkeypatch: pytest.MonkeyPatch,
    resolution: CameraResolution,
    cap: int,
    expected: int,
) -> None:
    """``decode_framerate`` is the camera rate capped at ``MAX_DECODE_FPS``.

    ``framerate`` stays the camera's native rate, so non-V4L2 sources are unaffected.
    """
    monkeypatch.setattr(ms, "MAX_DECODE_FPS", cap)
    server = _make_server(resolution)

    assert server.framerate == resolution.value[2]
    assert server.decode_framerate == expected


def test_max_decode_fps_env_override() -> None:
    """``REACHY_MINI_MAX_DECODE_FPS`` is read once at import time.

    Checked in a subprocess so the module under test is imported fresh with
    the variable set, without reloading it in this interpreter.
    """
    env = dict(os.environ, REACHY_MINI_MAX_DECODE_FPS="4")
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "from reachy_mini.media import media_server as m; print(m.MAX_DECODE_FPS)",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert out.stdout.strip() == "4"


def _framerate_of(capsfilter: ms.Gst.Element) -> tuple[int, int]:
    """Return the ``framerate`` fraction of a capsfilter's caps as (num, den)."""
    caps = capsfilter.get_property("caps")
    ok, num, den = caps.get_structure(0).get_fraction("framerate")
    assert ok, "capsfilter has no framerate"
    return num, den


@pytest.mark.parametrize(
    ("resolution", "native_fps"),
    [
        # 60 fps camera: drops to the cap before jpegdec.
        (CameraResolution.R1920x1080at60fps, 60),
        # 10 fps camera under a 10 fps cap: both capsfilters agree, nothing is dropped.
        (CameraResolution.R3840x2160at10fps, 10),
    ],
)
def test_v4l2_source_drops_frames_before_decoding(
    monkeypatch: pytest.MonkeyPatch, resolution: CameraResolution, native_fps: int
) -> None:
    """The V4L2 chain asks the camera for its native rate and drops to the cap before jpegdec."""
    ms.Gst.init([])
    if ms.Gst.ElementFactory.find("v4l2src") is None:
        pytest.skip("v4l2src plugin not available on this platform")

    monkeypatch.setattr(ms, "MAX_DECODE_FPS", 10)
    server = _make_server(resolution)

    elements = server._build_v4l2_source("/dev/video-does-not-exist")
    factories = [e.get_factory().get_name() for e in elements]

    assert factories == [
        "v4l2src",
        "capsfilter",
        "videorate",
        "capsfilter",
        "queue",
        "jpegdec",
        "videoconvert",
    ]
    camsrc, caps_native, videorate, caps_rate = elements[:4]

    assert camsrc.get_property("device") == "/dev/video-does-not-exist"
    # The camera negotiates what it actually offers...
    assert _framerate_of(caps_native) == (native_fps, 1)
    # ...and only the capped rate reaches the decoder, by dropping, never duplicating.
    assert videorate.get_property("drop-only") is True
    assert _framerate_of(caps_rate) == (10, 1)
