"""Tests for the media server's head-tracking video branch."""

import logging
import sys
from importlib.machinery import ModuleSpec
from threading import Event, Lock, Thread
from types import ModuleType
from unittest.mock import MagicMock

import gi
import numpy as np
import numpy.typing as npt
import pytest

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from reachy_mini.media import media_server as media_server_module  # noqa: E402
from reachy_mini.media.media_server import (  # noqa: E402
    TRACKER_FPS,
    TRACKER_HEIGHT,
    TRACKER_WIDTH,
    GstMediaServer,
)

Gst.init([])


def _tracking_branch_elements_available() -> bool:
    """Return whether the GStreamer elements used by the tracking branch exist."""
    return all(
        Gst.ElementFactory.find(name) is not None
        for name in (
            "queue",
            "valve",
            "videorate",
            "videoscale",
            "videoconvert",
            "capsfilter",
            "appsink",
            "tee",
        )
    )


def _required_element(pipeline: Gst.Pipeline, name: str) -> Gst.Element:
    """Return a named pipeline element, failing the test if it is absent."""
    element = pipeline.get_by_name(name)
    assert element is not None
    return element


@pytest.mark.skipif(
    not _tracking_branch_elements_available(),
    reason="GStreamer tracking branch elements are not installed",
)
def test_tracking_branch_starts_dropped_without_blocking_preroll() -> None:
    """The disabled tracking branch must not block WebRTC producer startup."""
    server = GstMediaServer.__new__(GstMediaServer)
    server._logger = logging.getLogger("test_media_server_tracking_branch")
    server._loop = MagicMock()
    server._bus_sender = MagicMock()
    server._tracking_lock = Lock()
    server._tracking_thread = None
    server._tracking_stop = Event()
    server._tracking_callback = None
    server._head_tracker = None

    pipeline = Gst.Pipeline.new("tracking-branch-test")
    tee = Gst.ElementFactory.make("tee", "tracking_test_tee")
    assert tee is not None
    pipeline.add(tee)

    server._build_tracking_branch(tee, pipeline)

    valve = _required_element(pipeline, "tracking_valve")
    appsink = _required_element(pipeline, "tracking_appsink")
    capsfilter = _required_element(pipeline, "tracking_capsfilter")

    assert valve.get_property("drop") is True
    assert valve.get_property("drop-mode").value_nick == "transform-to-gap"
    assert appsink.get_property("async") is False
    assert appsink.get_property("sync") is False
    assert capsfilter.get_property("caps").to_string() == (
        f"video/x-raw, format=(string)RGB, width=(int){TRACKER_WIDTH}, "
        f"height=(int){TRACKER_HEIGHT}, framerate=(fraction){TRACKER_FPS}/1"
    )


@pytest.mark.skipif(
    not _tracking_branch_elements_available(),
    reason="GStreamer tracking branch elements are not installed",
)
def test_enable_tracking_initializes_detector_off_command_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Head tracking startup must not block the SDK command handler."""
    constructor_started = Event()
    release_constructor = Event()
    enable_returned = Event()
    tracker_closed = Event()

    class BlockingHeadTracker:
        def __init__(self) -> None:
            constructor_started.set()
            release_constructor.wait(timeout=5.0)

        def get_head_position(
            self, img: npt.NDArray[np.uint8]
        ) -> tuple[npt.NDArray[np.float64], float] | tuple[None, None]:
            return None, None

        def close(self) -> None:
            tracker_closed.set()

    def fake_find_spec(name: str) -> ModuleSpec | None:
        if name == "mediapipe":
            return ModuleSpec(name, loader=None)
        return None

    head_tracker_module = ModuleType("reachy_mini.vision.head_tracker")
    head_tracker_module.HeadTracker = BlockingHeadTracker
    monkeypatch.setitem(
        sys.modules, "reachy_mini.vision.head_tracker", head_tracker_module
    )
    monkeypatch.setattr(media_server_module, "find_spec", fake_find_spec)

    server = GstMediaServer.__new__(GstMediaServer)
    server._logger = logging.getLogger("test_media_server_tracking_branch")
    server._loop = MagicMock()
    server._bus_sender = MagicMock()
    server._tracking_lock = Lock()
    server._tracking_thread = None
    server._tracking_stop = Event()
    server._tracking_callback = None
    server._head_tracker = None
    server._last_tracking_error_log = 0.0
    server._tracking_valve = Gst.ElementFactory.make("valve", "async_tracking_valve")
    server._tracking_appsink = Gst.ElementFactory.make(
        "appsink", "async_tracking_appsink"
    )
    assert server._tracking_valve is not None
    assert server._tracking_appsink is not None

    result: list[bool] = []

    def callback(
        eye_center: npt.NDArray[np.float64] | None,
        roll: float | None,
        width: int,
        height: int,
        camera_matrix: npt.NDArray[np.float64],
        distortion: npt.NDArray[np.float64],
        timestamp: float,
    ) -> None:
        pass

    def enable() -> None:
        result.append(server.enable_tracking(callback))
        enable_returned.set()

    enable_thread = Thread(target=enable, daemon=True)
    enable_thread.start()

    assert constructor_started.wait(timeout=1.0)
    assert enable_returned.wait(timeout=0.5)
    assert result == [True]

    release_constructor.set()
    server.disable_tracking()
    enable_thread.join(timeout=1.0)
    assert tracker_closed.wait(timeout=1.0)
