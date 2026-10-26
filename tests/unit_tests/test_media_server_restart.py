"""Unit tests for restarting the WebRTC pipeline after an error.

An ERROR on the sender pipeline stops it (video and audio together) while
WebRTC clients can still connect, so the media server rebuilds it like
`stop()` / `start()` do, within a restart budget. GLib timers and the
pipeline itself are stubbed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, cast
from unittest.mock import MagicMock

import pytest

from reachy_mini.media import media_server as ms
from reachy_mini.media.media_server import (
    PIPELINE_RESTART_DELAY_S,
    PIPELINE_RESTART_LIMIT,
    PIPELINE_RESTART_WINDOW_S,
    GstMediaServer,
)


def _make_server(running: bool = True) -> GstMediaServer:
    """Build a minimal ``GstMediaServer`` instance without booting GStreamer."""
    server = cast(GstMediaServer, object.__new__(GstMediaServer))
    server._logger = logging.getLogger("test_restart")
    server._mic_reopen_source_id = None
    server._pipeline_running = running
    server._pipeline_restart_source_id = None
    server._pipeline_restart_times = []
    server._pipeline_sender = MagicMock()
    # Stubs for `__del__` -> `close()`.
    server._loop = MagicMock()
    server._bus_sender = MagicMock()
    return server


@pytest.fixture
def timers(monkeypatch: pytest.MonkeyPatch) -> Dict[str, List[Any]]:
    """Replace ``GLib.timeout_add_seconds`` / ``source_remove`` with stubs."""
    journal: Dict[str, List[Any]] = {"added": [], "removed": []}
    next_id = [200]

    def fake_add(secs: int, fn: Any, *args: Any) -> int:
        next_id[0] += 1
        journal["added"].append((next_id[0], secs, fn, args))
        return next_id[0]

    monkeypatch.setattr(ms.GLib, "timeout_add_seconds", fake_add)
    monkeypatch.setattr(
        ms.GLib, "source_remove", lambda sid: journal["removed"].append(sid)
    )
    return journal


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> List[float]:
    now = [1000.0]
    monkeypatch.setattr(ms.time, "monotonic", lambda: now[0])
    return now


def _error(server: GstMediaServer, pipeline: Any = None) -> bool:
    msg = MagicMock(type=ms.Gst.MessageType.ERROR)
    msg.parse_error.return_value = (MagicMock(message="boom"), "")
    target = server._pipeline_sender if pipeline is None else pipeline
    return server._on_bus_message(MagicMock(), msg, target)


def test_error_on_a_running_pipeline_schedules_one_restart(
    timers: Dict[str, List[Any]], clock: List[float]
) -> None:
    server = _make_server()

    assert _error(server) is False  # the dead pipeline's watch goes away
    _error(server)  # already pending

    assert len(timers["added"]) == 1
    _, delay_s, fn, _ = timers["added"][0]
    assert delay_s == PIPELINE_RESTART_DELAY_S
    assert fn == server._restart_pipeline


def test_no_restart_when_stopped_or_for_a_stale_pipeline(
    timers: Dict[str, List[Any]], clock: List[float]
) -> None:
    _error(_make_server(running=False))
    _error(_make_server(), pipeline=MagicMock())

    assert timers["added"] == []


def test_restart_stops_then_starts(
    monkeypatch: pytest.MonkeyPatch, timers: Dict[str, List[Any]], clock: List[float]
) -> None:
    server = _make_server()
    calls: List[str] = []
    monkeypatch.setattr(server, "stop", lambda: calls.append("stop"))
    monkeypatch.setattr(server, "start", lambda: calls.append("start"))

    _error(server)
    assert server._restart_pipeline() is False

    assert calls == ["stop", "start"]
    assert server._pipeline_restart_source_id is None


def test_restarts_are_capped_per_window(
    monkeypatch: pytest.MonkeyPatch, timers: Dict[str, List[Any]], clock: List[float]
) -> None:
    server = _make_server()
    monkeypatch.setattr(server, "stop", lambda: None)
    monkeypatch.setattr(server, "start", lambda: None)

    for _ in range(PIPELINE_RESTART_LIMIT + 2):
        _error(server)
        if server._pipeline_restart_source_id is not None:
            server._restart_pipeline()
        clock[0] += 1.0
    assert len(timers["added"]) == PIPELINE_RESTART_LIMIT

    clock[0] += PIPELINE_RESTART_WINDOW_S
    _error(server)
    assert len(timers["added"]) == PIPELINE_RESTART_LIMIT + 1


def test_stop_cancels_a_pending_restart(
    timers: Dict[str, List[Any]], clock: List[float]
) -> None:
    server = _make_server()
    _error(server)
    pending = server._pipeline_restart_source_id

    server.stop()

    assert timers["removed"] == [pending]
    assert server._pipeline_restart_source_id is None
    assert server._pipeline_running is False
