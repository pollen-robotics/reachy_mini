"""Tests for the media server's head-tracking video branch."""

import logging
from threading import Event, Lock
from unittest.mock import MagicMock

import gi
import pytest

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

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
