"""Tests for media-server pipeline startup."""

from unittest.mock import MagicMock, call

import pytest

from reachy_mini.media import media_server
from reachy_mini.media.media_server import GstMediaServer, SimulationMode


def test_initial_start_reuses_pipeline_and_restart_rebuilds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The constructor pipeline is reused once, then rebuilt after stop."""
    build_pipeline = MagicMock()
    monkeypatch.setattr(GstMediaServer, "_build_pipeline", build_pipeline)
    monkeypatch.setattr(media_server, "Thread", MagicMock())
    monkeypatch.setattr(
        media_server.GLib,
        "timeout_add_seconds",
        MagicMock(),
    )
    server = GstMediaServer(sim_mode=SimulationMode.MUJOCO)
    server._bus_sender = MagicMock()
    server._pipeline_sender = MagicMock()

    server.start()

    assert build_pipeline.call_count == 1

    server.stop()
    server.start()

    assert build_pipeline.call_count == 2
    server._bus_sender.remove_watch.assert_called_once_with()
    assert server._pipeline_sender.set_state.call_args_list == [
        call(media_server.Gst.State.PLAYING),
        call(media_server.Gst.State.NULL),
        call(media_server.Gst.State.PLAYING),
    ]
    server.close()
