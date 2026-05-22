"""Unit tests for the WebRTC negotiation watchdog.

The watchdog (in :mod:`reachy_mini.media.media_server`) is the
daemon-side fix for the historical "stuck at session" bug, where
libnice (used by webrtcbin internally) freezes mid-CHECKING and
the JS client spins forever waiting on an offer that will never
complete. These tests focus on the pure logic of the watchdog:
state tracking, deadline handling, and the failure notification
fan-out. The webrtcbin / GLib edges are mocked so the suite runs
without a live GStreamer pipeline.

Out of scope (would belong to an integration test):

* End-to-end signal connection (``notify::*-state``) firing on a
  real ``webrtcbin``.
* Real ``GLib`` timer scheduling.
* Cross-thread interaction with the central signaling relay.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any, Dict, List, Tuple, cast
from unittest.mock import MagicMock

import pytest

from reachy_mini.media.media_server import (
    ICE_NEGOTIATION_DEADLINE_S,
    SESSION_FAILED_REASON_ICE_TIMEOUT,
    SESSION_FAILED_REASON_PC_FAILED,
    GstMediaServer,
    _PeerWebRTCState,
)

# ---------------------------------------------------------------------------
# _PeerWebRTCState helpers
# ---------------------------------------------------------------------------


def test_peer_state_default_is_all_new() -> None:
    """A freshly-created state defaults all three webrtcbin states to ``"new"``."""
    state = _PeerWebRTCState(peer_id="peer-1")

    assert state.peer_id == "peer-1"
    assert state.ice_state == "new"
    assert state.conn_state == "new"
    assert state.signaling_state == "new"
    assert state.failure_notified is False
    assert state.watchdog_source_id is None


def test_peer_state_asdict_round_trip() -> None:
    """``asdict`` returns the exact set of fields downstream code reads."""
    state = _PeerWebRTCState(peer_id="peer-2")
    state.ice_state = "checking"
    state.conn_state = "connecting"
    state.signaling_state = "have-local-offer"

    snapshot = state.asdict()

    assert set(snapshot.keys()) == {
        "peer_id",
        "ice_state",
        "conn_state",
        "signaling_state",
        "elapsed_s",
    }
    assert snapshot["peer_id"] == "peer-2"
    assert snapshot["ice_state"] == "checking"
    assert snapshot["conn_state"] == "connecting"
    assert snapshot["signaling_state"] == "have-local-offer"
    assert isinstance(snapshot["elapsed_s"], float)


def test_reason_constants_are_distinct_strings() -> None:
    """The wire-level reason constants must be distinct strings."""
    assert SESSION_FAILED_REASON_ICE_TIMEOUT != SESSION_FAILED_REASON_PC_FAILED
    assert isinstance(SESSION_FAILED_REASON_ICE_TIMEOUT, str)
    assert isinstance(SESSION_FAILED_REASON_PC_FAILED, str)


def test_ice_negotiation_deadline_is_reasonable() -> None:
    """The deadline constant sits inside a sane safety envelope."""
    # A sub-2s deadline would false-positive on slow networks; a
    # >30s deadline defeats the UX purpose (user has already given up).
    assert 2 < ICE_NEGOTIATION_DEADLINE_S < 30


# ---------------------------------------------------------------------------
# Bare-bones stand-in for GstMediaServer
# ---------------------------------------------------------------------------
#
# `GstMediaServer.__init__` calls `Gst.init([])` and starts a
# pipeline, which is overkill for testing the watchdog in isolation.
# We bypass `__init__` and only initialise the attributes the
# watchdog code touches; same approach as the existing
# `test_audio_gstreamer.py`.


def _make_server() -> GstMediaServer:
    """Build a minimal ``GstMediaServer`` instance without booting GStreamer."""
    server = cast(GstMediaServer, object.__new__(GstMediaServer))
    server._logger = logging.getLogger("test_watchdog")
    server._peer_states = {}
    server._peer_states_lock = Lock()
    server._on_session_failed = None
    # Stubs for `__del__` -> `close()`, which the destructor calls
    # at GC time; without them we'd surface a noisy
    # ``AttributeError`` ``unraisable`` warning during the test
    # session teardown. Real instances have these set in
    # ``__init__``; we're bypassing it on purpose.
    server._loop = MagicMock()
    server._bus_sender = MagicMock()
    return server


def _patch_glib(monkeypatch: pytest.MonkeyPatch) -> Dict[str, List[Any]]:
    """Replace ``GLib.timeout_add_seconds`` / ``source_remove`` with stubs."""
    journal: Dict[str, List[Any]] = {"added": [], "removed": []}
    next_id = [1000]

    def fake_add(secs: int, fn: Any, *args: Any) -> int:
        source_id = next_id[0]
        next_id[0] += 1
        journal["added"].append((source_id, secs, fn, args))
        return source_id

    def fake_remove(source_id: int) -> bool:
        journal["removed"].append(source_id)
        return True

    from reachy_mini.media import media_server as ms

    monkeypatch.setattr(ms.GLib, "timeout_add_seconds", fake_add)
    monkeypatch.setattr(ms.GLib, "source_remove", fake_remove)
    return journal


# ---------------------------------------------------------------------------
# _install_negotiation_watchdog
# ---------------------------------------------------------------------------


def test_install_watchdog_tracks_state_and_schedules_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A freshly-added peer is tracked, signalled, and timer-armed."""
    journal = _patch_glib(monkeypatch)
    server = _make_server()
    webrtcbin = MagicMock()

    server._install_negotiation_watchdog("peer-1", webrtcbin)

    assert "peer-1" in server._peer_states
    state = server._peer_states["peer-1"]
    assert state.peer_id == "peer-1"
    assert state.watchdog_source_id == 1000

    connected_signals = {call.args[0] for call in webrtcbin.connect.call_args_list}
    assert connected_signals == {
        "notify::ice-connection-state",
        "notify::connection-state",
        "notify::signaling-state",
    }

    assert len(journal["added"]) == 1
    _source_id, secs, _fn, args = journal["added"][0]
    assert secs == ICE_NEGOTIATION_DEADLINE_S
    assert args == ("peer-1",)


def test_install_watchdog_replaces_pre_existing_state_for_same_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A re-fired ``consumer-added`` for the same peer cancels the old timer.

    Some webrtcsink builds re-fire ``consumer-added`` after a brief
    disconnect; we must not stack watchdogs that would all fire.
    """
    journal = _patch_glib(monkeypatch)
    server = _make_server()
    webrtcbin = MagicMock()

    server._install_negotiation_watchdog("peer-1", webrtcbin)
    first_source_id = server._peer_states["peer-1"].watchdog_source_id

    server._install_negotiation_watchdog("peer-1", webrtcbin)

    assert first_source_id in journal["removed"], (
        "the previous watchdog timer must be cancelled before the new one fires"
    )
    new_state = server._peer_states["peer-1"]
    assert new_state.watchdog_source_id != first_source_id


# ---------------------------------------------------------------------------
# _teardown_negotiation_watchdog
# ---------------------------------------------------------------------------


def test_teardown_watchdog_drops_state_and_cancels_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_consumer_removed`` clears both the state dict and the GLib source."""
    journal = _patch_glib(monkeypatch)
    server = _make_server()
    webrtcbin = MagicMock()
    server._install_negotiation_watchdog("peer-1", webrtcbin)

    server._teardown_negotiation_watchdog("peer-1")

    assert "peer-1" not in server._peer_states
    assert journal["removed"] == [1000]


def test_teardown_watchdog_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling teardown twice must be a clean no-op."""
    _patch_glib(monkeypatch)
    server = _make_server()
    webrtcbin = MagicMock()
    server._install_negotiation_watchdog("peer-1", webrtcbin)

    server._teardown_negotiation_watchdog("peer-1")
    server._teardown_negotiation_watchdog("peer-1")  # must not raise


# ---------------------------------------------------------------------------
# State change handlers
# ---------------------------------------------------------------------------


def _fake_webrtcbin_with_state(prop_to_nick: Dict[str, str]) -> MagicMock:
    """Build a webrtcbin mock whose ``get_property`` returns the chosen nick."""
    webrtcbin = MagicMock()

    def get_property(prop: str) -> Any:
        nick = prop_to_nick.get(prop, "unknown")
        return MagicMock(value_nick=nick)

    webrtcbin.get_property.side_effect = get_property
    return webrtcbin


def test_ice_state_change_updates_tracked_state() -> None:
    """A ``notify::ice-connection-state`` callback updates the tracked state."""
    server = _make_server()
    server._peer_states["peer-1"] = _PeerWebRTCState(peer_id="peer-1")
    webrtcbin = _fake_webrtcbin_with_state({"ice-connection-state": "checking"})

    server._on_ice_connection_state_change(webrtcbin, None, "peer-1")

    assert server._peer_states["peer-1"].ice_state == "checking"


def test_connection_state_failed_triggers_pc_failed_notification() -> None:
    """``connection-state == failed`` notifies eagerly without waiting."""
    received: List[Tuple[str, str, Dict[str, Any]]] = []
    server = _make_server()
    server._on_session_failed = lambda *args: received.append(args)
    server._peer_states["peer-1"] = _PeerWebRTCState(peer_id="peer-1")
    webrtcbin = _fake_webrtcbin_with_state({"connection-state": "failed"})

    server._on_connection_state_change(webrtcbin, None, "peer-1")

    assert len(received) == 1
    peer_id, reason, _diag = received[0]
    assert peer_id == "peer-1"
    assert reason == SESSION_FAILED_REASON_PC_FAILED
    # State was dropped after the notify path teared the watchdog down.
    assert "peer-1" not in server._peer_states


def test_connection_state_connecting_does_not_notify() -> None:
    """Intermediate states like ``connecting`` must not trip the failure callback."""
    received: List[Any] = []
    server = _make_server()
    server._on_session_failed = lambda *args: received.append(args)
    server._peer_states["peer-1"] = _PeerWebRTCState(peer_id="peer-1")
    webrtcbin = _fake_webrtcbin_with_state({"connection-state": "connecting"})

    server._on_connection_state_change(webrtcbin, None, "peer-1")

    assert received == []
    assert server._peer_states["peer-1"].conn_state == "connecting"


def test_connection_state_failed_only_fires_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated ``connection-state == failed`` must only emit one ``endSession``."""
    _patch_glib(monkeypatch)
    received: List[Tuple[str, str, Dict[str, Any]]] = []
    server = _make_server()
    server._on_session_failed = lambda *args: received.append(args)
    server._peer_states["peer-1"] = _PeerWebRTCState(peer_id="peer-1")
    webrtcbin = _fake_webrtcbin_with_state({"connection-state": "failed"})

    server._on_connection_state_change(webrtcbin, None, "peer-1")
    # Re-add stub state to simulate a second `failed` firing in a
    # world where teardown is async.
    server._peer_states["peer-1"] = _PeerWebRTCState(peer_id="peer-1")
    server._peer_states["peer-1"].failure_notified = True
    server._on_connection_state_change(webrtcbin, None, "peer-1")

    assert len(received) == 1


# ---------------------------------------------------------------------------
# _on_negotiation_deadline
# ---------------------------------------------------------------------------


def test_deadline_fires_session_failed_when_stuck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A peer still ``checking`` past the deadline reports ``ICE_TIMEOUT``."""
    _patch_glib(monkeypatch)
    received: List[Tuple[str, str, Dict[str, Any]]] = []
    server = _make_server()
    server._on_session_failed = lambda *args: received.append(args)
    state = _PeerWebRTCState(peer_id="peer-1")
    state.ice_state = "checking"
    state.conn_state = "new"
    state.watchdog_source_id = 999
    server._peer_states["peer-1"] = state

    keep = server._on_negotiation_deadline("peer-1")

    assert keep is False, "GLib timer must not be re-scheduled"
    assert len(received) == 1
    peer_id, reason, diag = received[0]
    assert peer_id == "peer-1"
    assert reason == SESSION_FAILED_REASON_ICE_TIMEOUT
    assert diag["ice_state"] == "checking"
    assert "peer-1" not in server._peer_states


def test_deadline_skips_when_connection_is_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deadline firing on a connected peer must NOT report a failure."""
    _patch_glib(monkeypatch)
    received: List[Any] = []
    server = _make_server()
    server._on_session_failed = lambda *args: received.append(args)
    state = _PeerWebRTCState(peer_id="peer-1")
    state.ice_state = "connected"
    state.conn_state = "connected"
    state.watchdog_source_id = 999
    server._peer_states["peer-1"] = state

    keep = server._on_negotiation_deadline("peer-1")

    assert keep is False
    assert received == []
    # Still tracked: `_consumer_removed` is the canonical cleanup path.
    assert "peer-1" in server._peer_states
    assert server._peer_states["peer-1"].watchdog_source_id is None


def test_deadline_no_op_after_failure_already_notified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deadline firing after eager-notify must not double-fire the callback."""
    _patch_glib(monkeypatch)
    received: List[Any] = []
    server = _make_server()
    server._on_session_failed = lambda *args: received.append(args)
    state = _PeerWebRTCState(peer_id="peer-1")
    state.failure_notified = True
    state.watchdog_source_id = 999
    server._peer_states["peer-1"] = state

    server._on_negotiation_deadline("peer-1")

    assert received == []


def test_deadline_for_unknown_peer_is_no_op() -> None:
    """A timer firing after ``_consumer_removed`` already ran is a clean no-op."""
    server = _make_server()
    keep = server._on_negotiation_deadline("ghost-peer")
    assert keep is False


# ---------------------------------------------------------------------------
# _dispatch_session_failed
# ---------------------------------------------------------------------------


def test_dispatch_swallows_handler_exceptions() -> None:
    """A misbehaving handler must not crash the watchdog or the bus thread."""
    server = _make_server()

    def bad_handler(*_args: Any) -> None:
        raise RuntimeError("downstream is on fire")

    server._on_session_failed = bad_handler

    server._dispatch_session_failed("peer-1", "any-reason", {"foo": "bar"})


def test_dispatch_no_op_when_handler_unwired() -> None:
    """Without a wired handler, dispatching is a clean no-op."""
    server = _make_server()
    server._dispatch_session_failed("peer-1", "any-reason", {"foo": "bar"})


# ---------------------------------------------------------------------------
# set_session_failed_handler wiring
# ---------------------------------------------------------------------------


def test_set_session_failed_handler_records_callback() -> None:
    """The setter is the daemon's wiring entry point for the relay handler."""
    server = _make_server()
    captured: List[Tuple[str, str, Dict[str, Any]]] = []

    def handler(peer_id: str, reason: str, diag: Dict[str, Any]) -> None:
        captured.append((peer_id, reason, diag))

    server.set_session_failed_handler(handler)
    server._dispatch_session_failed("peer-9", "x", {"a": 1})

    assert captured == [("peer-9", "x", {"a": 1})]
