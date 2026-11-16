"""Per-session liveness of the peer-to-peer link in the media server.

The relay decides whether a session is alive from its own link, keyed by
the signalling session id webrtcsink puts in each webrtcbin's name (the
consumer peer id is shared by every relayed session). A client counts as
alive while it sends RTCP - which any client receiving media does, even
one that never writes on the data channel - or data-channel messages, and
as gone once it sent an RTCP BYE.
"""

from __future__ import annotations

import logging
import math
import time
import types
from threading import Lock

import pytest

from reachy_mini.media.media_server import GstMediaServer, _session_id_of


def _server() -> GstMediaServer:
    """Bare instance carrying only the liveness state (no pipeline)."""
    server = GstMediaServer.__new__(GstMediaServer)
    server._session_activity = {}
    server._session_bye = set()
    server._session_rtcp_seen = set()
    server._session_activity_lock = Lock()
    server._logger = logging.getLogger("test")
    server.close = lambda: None  # type: ignore[method-assign]  # nothing to tear down
    return server


def _bin(name: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(get_name=lambda: name)


def test_session_id_comes_from_the_webrtcbin_name() -> None:
    """Webrtcsink's `webrtcbin-<session id>` names give the session key."""
    assert _session_id_of(_bin("webrtcbin-6485b7ad-8146")) == "6485b7ad-8146"  # type: ignore[arg-type]
    assert _session_id_of(_bin("webrtcbin0")) is None  # type: ignore[arg-type]
    assert _session_id_of(_bin("webrtcbin-")) is None  # type: ignore[arg-type]


def test_untracked_session_is_not_judged() -> None:
    """An unknown session reports None, never a fake idle time."""
    assert _server().session_idle_seconds("ghost") is None


def test_rtcp_and_data_keep_a_session_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """RTCP from a receive-only client counts, like data-channel messages."""
    server = _server()
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    server._session_activity["s1"] = now[0]

    now[0] += 20.0
    assert server.session_idle_seconds("s1") == pytest.approx(20.0)

    server._on_session_rtcp(None, 0, 1234, "s1")  # type: ignore[arg-type]
    assert server.session_idle_seconds("s1") == pytest.approx(0.0)

    now[0] += 3.0
    server._on_session_dc_message(None, "{}", "s1")  # type: ignore[arg-type]
    assert server.session_idle_seconds("s1") == pytest.approx(0.0)


def test_bye_marks_the_session_gone_and_removal_forgets_it() -> None:
    """An RTCP BYE means the client hung up; removal drops the entry."""
    server = _server()
    server._session_activity["s1"] = time.monotonic()

    server._on_session_bye(None, 0, 1234, "s1")  # type: ignore[arg-type]
    assert server.session_idle_seconds("s1") == math.inf

    server._forget_session_liveness("s1")
    assert server.session_idle_seconds("s1") is None


def test_signs_of_life_for_unknown_sessions_are_ignored() -> None:
    """A late RTCP after removal must not resurrect the session."""
    server = _server()
    server._on_session_rtcp(None, 0, 1, "gone")  # type: ignore[arg-type]
    server._on_session_bye(None, 0, 1, "gone")  # type: ignore[arg-type]
    assert server.session_idle_seconds("gone") is None
