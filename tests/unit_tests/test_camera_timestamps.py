"""Tests for camera frame timestamp handoff across daemon local IPC."""

from types import SimpleNamespace

import gi
import pytest

from reachy_mini.media.camera_timestamps import CameraFrameTimestamps
from reachy_mini.media.media_server import GstMediaServer

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402


def test_source_pts_is_converted_to_monotonic_capture_time() -> None:
    """Producer pipeline age is subtracted from the local handoff instant."""
    timestamps = CameraFrameTimestamps()

    selected = timestamps.record(
        42,
        100.0,
        frame_running_time_s=9.8,
        pipeline_running_time_s=10.0,
    )

    assert selected == pytest.approx(99.8)
    assert timestamps.pop(42) == pytest.approx(99.8)
    assert timestamps.pop(42) is None


def test_invalid_source_age_falls_back_to_pre_ipc_handoff() -> None:
    """Missing, future, or implausibly old PTS never creates a bogus clock."""
    timestamps = CameraFrameTimestamps(max_source_age_s=2.0)

    assert timestamps.record(1, 100.0) == 100.0
    assert (
        timestamps.record(
            2, 100.0, frame_running_time_s=10.1, pipeline_running_time_s=10.0
        )
        == 100.0
    )
    assert (
        timestamps.record(
            3, 100.0, frame_running_time_s=1.0, pipeline_running_time_s=10.0
        )
        == 100.0
    )


def test_registry_is_bounded_and_clearable() -> None:
    """Skipped camera consumers cannot grow the producer registry forever."""
    timestamps = CameraFrameTimestamps(capacity=2)
    timestamps.record(1, 1.0)
    timestamps.record(2, 2.0)
    timestamps.record(3, 3.0)

    assert timestamps.pop(1) is None
    assert timestamps.pop(2) == 2.0
    timestamps.clear()
    assert timestamps.pop(3) is None


def test_media_server_converts_producer_pts_through_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual IPC pad callback maps segment PTS into monotonic time."""
    registry = CameraFrameTimestamps()
    server = SimpleNamespace(_camera_frame_timestamps=registry)
    buffer = SimpleNamespace(offset=42, pts=int(9.8 * Gst.SECOND))
    info = SimpleNamespace(get_buffer=lambda: buffer)
    segment = SimpleNamespace(
        to_running_time=lambda _format, pts: pts,
    )
    event = SimpleNamespace(parse_segment=lambda: segment)
    pad = SimpleNamespace(get_sticky_event=lambda _event_type, _index: event)
    pipeline = SimpleNamespace(get_current_running_time=lambda: int(10.0 * Gst.SECOND))
    monkeypatch.setattr("reachy_mini.media.media_server.time.monotonic", lambda: 100.0)

    result = GstMediaServer._record_camera_frame_timestamp(
        server,  # type: ignore[arg-type]
        pad,  # type: ignore[arg-type]
        info,  # type: ignore[arg-type]
        pipeline,  # type: ignore[arg-type]
    )

    assert result == int(Gst.PadProbeReturn.OK)
    assert registry.pop(42) == pytest.approx(99.8)
