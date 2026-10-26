"""Unit tests for reopening the Windows mic after a device failure.

With ``continue-on-error`` set, ``wasapi2src`` posts a warning instead of an
error when Windows invalidates its endpoint, and keeps the pipeline (video
included) running on silence. The media server then reopens the device by
setting ``device`` again. GLib timers are stubbed so the suite runs without a
main loop or audio hardware.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, cast
from unittest.mock import MagicMock

import pytest

from reachy_mini.media import media_server as ms
from reachy_mini.media.media_server import (
    MIC_REOPEN_DELAY_S,
    MIC_REOPEN_MAX_DELAY_S,
    MIC_REOPEN_RESET_S,
    GstMediaServer,
)


def _make_server() -> GstMediaServer:
    """Build a minimal ``GstMediaServer`` instance without booting GStreamer."""
    server = cast(GstMediaServer, object.__new__(GstMediaServer))
    server._logger = logging.getLogger("test_mic_recovery")
    server._mic_reopen_source_id = None
    server._mic_reopen_delay_s = MIC_REOPEN_DELAY_S
    server._mic_last_failure = 0.0
    server._pipeline_running = True
    server._pipeline_restart_source_id = None
    server._pipeline_restart_times = []
    server._pipeline_sender = MagicMock()
    # Stubs for `__del__` -> `close()`.
    server._loop = MagicMock()
    server._bus_sender = MagicMock()
    return server


@pytest.fixture
def timers(monkeypatch: pytest.MonkeyPatch) -> Dict[str, List[Any]]:
    """Replace ``GLib.timeout_add`` / ``source_remove`` with recording stubs."""
    journal: Dict[str, List[Any]] = {"added": [], "removed": []}
    next_id = [100]

    def fake_add(ms_delay: int, fn: Any, *args: Any) -> int:
        next_id[0] += 1
        journal["added"].append((next_id[0], ms_delay, fn, args))
        return next_id[0]

    monkeypatch.setattr(ms.GLib, "timeout_add", fake_add)
    monkeypatch.setattr(
        ms.GLib, "source_remove", lambda sid: journal["removed"].append(sid)
    )
    return journal


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> List[float]:
    """Controllable ``time.monotonic`` for the reopen back-off."""
    now = [1000.0]
    monkeypatch.setattr(ms.time, "monotonic", lambda: now[0])
    return now


def _fire(server: GstMediaServer, timers: Dict[str, List[Any]]) -> None:
    """Run the most recently scheduled reopen, as GLib would."""
    _, _, fn, args = timers["added"][-1]
    assert fn(*args) is False


def _warning(domain: int, code: Any) -> ms.Gst.Message:
    """A real WARNING message, as the mic element would post it."""
    err = ms.GLib.Error.new_literal(domain, "mic warning", code)
    return ms.Gst.Message.new_warning(ms.Gst.ElementFactory.make("fakesrc"), err, "")


# ---------------------------------------------------------------------------
# Bus routing
# ---------------------------------------------------------------------------


def test_is_wasapi2src_rejects_other_sources() -> None:
    assert not ms._is_wasapi2src(None)
    assert not ms._is_wasapi2src(ms.Gst.ElementFactory.make("fakesrc"))


def test_mic_warning_schedules_a_reopen_instead_of_the_default_handler(
    monkeypatch: pytest.MonkeyPatch, timers: Dict[str, List[Any]], clock: List[float]
) -> None:
    server = _make_server()
    monkeypatch.setattr(ms, "_is_wasapi2src", lambda obj: True)
    default = MagicMock(return_value=False)
    monkeypatch.setattr(ms, "handle_default_bus_message", default)
    msg = _warning(ms.Gst.ResourceError.quark(), ms.Gst.ResourceError.READ)

    assert server._on_bus_message(MagicMock(), msg, server._pipeline_sender) is True
    default.assert_not_called()
    assert len(timers["added"]) == 1


def test_mic_overrun_warning_does_not_reopen(
    monkeypatch: pytest.MonkeyPatch, timers: Dict[str, List[Any]]
) -> None:
    server = _make_server()
    monkeypatch.setattr(ms, "_is_wasapi2src", lambda obj: True)
    default = MagicMock(return_value=True)
    monkeypatch.setattr(ms, "handle_default_bus_message", default)
    msg = _warning(ms.Gst.CoreError.quark(), ms.Gst.CoreError.CLOCK)

    assert server._on_bus_message(MagicMock(), msg, server._pipeline_sender) is True
    default.assert_called_once()
    assert timers["added"] == []


def test_other_messages_go_to_the_default_handler(
    monkeypatch: pytest.MonkeyPatch, timers: Dict[str, List[Any]]
) -> None:
    server = _make_server()
    monkeypatch.setattr(ms, "_is_wasapi2src", lambda obj: False)
    default = MagicMock(return_value=True)
    monkeypatch.setattr(ms, "handle_default_bus_message", default)
    msg = MagicMock(type=ms.Gst.MessageType.WARNING)

    assert server._on_bus_message(MagicMock(), msg, server._pipeline_sender) is True
    default.assert_called_once()
    assert timers["added"] == []


# ---------------------------------------------------------------------------
# Reopen scheduling and back-off
# ---------------------------------------------------------------------------


def test_failure_schedules_one_reopen_at_the_base_delay(
    timers: Dict[str, List[Any]], clock: List[float]
) -> None:
    server = _make_server()
    audiosrc = MagicMock()

    server._on_mic_failure(audiosrc, "invalidated")
    server._on_mic_failure(audiosrc, "invalidated")  # already pending

    assert len(timers["added"]) == 1
    _, delay_ms, fn, args = timers["added"][0]
    assert delay_ms == int(MIC_REOPEN_DELAY_S * 1000)
    assert fn == server._reopen_mic and args == (audiosrc,)


def test_repeated_failures_back_off_up_to_the_cap(
    timers: Dict[str, List[Any]], clock: List[float]
) -> None:
    server = _make_server()
    audiosrc = MagicMock()
    audiosrc.get_parent.return_value = server._pipeline_sender

    delays = []
    for _ in range(8):
        server._on_mic_failure(audiosrc, "invalidated")
        delays.append(timers["added"][-1][1] / 1000)
        _fire(server, timers)
        clock[0] += 1.0

    assert delays[:3] == [
        MIC_REOPEN_DELAY_S,
        2 * MIC_REOPEN_DELAY_S,
        4 * MIC_REOPEN_DELAY_S,
    ]
    assert max(delays) == MIC_REOPEN_MAX_DELAY_S


def test_back_off_resets_after_a_quiet_spell(
    timers: Dict[str, List[Any]], clock: List[float]
) -> None:
    server = _make_server()
    audiosrc = MagicMock()
    for _ in range(3):
        server._on_mic_failure(audiosrc, "invalidated")
        _fire(server, timers)

    clock[0] += MIC_REOPEN_RESET_S + 1
    server._on_mic_failure(audiosrc, "invalidated")
    assert timers["added"][-1][1] == int(MIC_REOPEN_DELAY_S * 1000)


# ---------------------------------------------------------------------------
# Reopen itself
# ---------------------------------------------------------------------------


def test_reopen_sets_the_same_device_again(
    timers: Dict[str, List[Any]], clock: List[float]
) -> None:
    server = _make_server()
    audiosrc = MagicMock()
    audiosrc.get_parent.return_value = server._pipeline_sender
    audiosrc.get_property.return_value = "{0.0.1.00000000}.{mic}"

    server._on_mic_failure(audiosrc, "invalidated")
    _fire(server, timers)

    audiosrc.set_property.assert_called_once_with("device", "{0.0.1.00000000}.{mic}")
    assert server._mic_reopen_source_id is None


def test_reopen_skips_a_source_from_a_rebuilt_pipeline(
    timers: Dict[str, List[Any]], clock: List[float]
) -> None:
    server = _make_server()
    audiosrc = MagicMock()
    audiosrc.get_parent.return_value = MagicMock()  # an older pipeline

    server._on_mic_failure(audiosrc, "invalidated")
    _fire(server, timers)

    audiosrc.set_property.assert_not_called()


def test_stop_cancels_a_pending_reopen(
    timers: Dict[str, List[Any]], clock: List[float]
) -> None:
    server = _make_server()
    server._on_mic_failure(MagicMock(), "invalidated")
    pending = server._mic_reopen_source_id

    server.stop()

    assert timers["removed"] == [pending]
    assert server._mic_reopen_source_id is None


# ---------------------------------------------------------------------------
# Windows mic element (needs the wasapi2 plugin)
# ---------------------------------------------------------------------------


def _require_wasapi2() -> None:
    if ms.Gst.ElementFactory.find("wasapi2src") is None:
        pytest.skip("wasapi2src is only available on Windows")


def test_is_wasapi2src_accepts_the_windows_mic() -> None:
    _require_wasapi2()
    assert ms._is_wasapi2src(ms.Gst.ElementFactory.make("wasapi2src"))


def test_windows_mic_continues_on_device_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_wasapi2()
    server = _make_server()
    monkeypatch.setattr(ms, "has_reachymini_asoundrc", lambda: False)
    monkeypatch.setattr(ms, "get_audio_device", lambda kind: "{0.0.1.00000000}.{mic}")
    monkeypatch.setattr(ms.platform, "system", lambda: "Windows")

    audiosrc = server._build_audio_source()

    assert audiosrc is not None
    assert audiosrc.get_property("continue-on-error") is True
    assert audiosrc.get_property("device") == "{0.0.1.00000000}.{mic}"
