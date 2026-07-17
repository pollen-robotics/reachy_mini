"""Tests for the WebRTC signalling helpers.

`connect` is mocked with a scripted fake websocket, so the producer-list
request/parse logic is covered without a real signalling server.
"""

import json

import pytest

from reachy_mini.media import webrtc_utils


class _FakeWS:
    """Scripted stand-in for a websockets sync connection (context manager)."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.sent: list[str] = []

    def __enter__(self) -> "_FakeWS":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def recv(self) -> str:
        return self._replies.pop(0)

    def send(self, message: str) -> None:
        self.sent.append(message)


def _patch_connect(monkeypatch, ws: _FakeWS) -> None:
    monkeypatch.setattr(webrtc_utils, "connect", lambda uri: ws)


def test_get_producer_list_parses_producers(monkeypatch) -> None:
    """Welcome is ignored, a `list` request is sent, producers are decoded."""
    reply = json.dumps(
        {
            "type": "list",
            "producers": [
                {"id": "peer1", "meta": {"name": "reachy"}},
                {"id": "peer2", "meta": {"name": "other"}},
            ],
        }
    )
    ws = _FakeWS(["welcome", reply])
    _patch_connect(monkeypatch, ws)

    producers = webrtc_utils.get_producer_list("host", 8443)

    assert producers == {"peer1": {"name": "reachy"}, "peer2": {"name": "other"}}
    assert ws.sent == [json.dumps({"type": "list"})]


def test_get_producer_list_unknown_type_returns_empty(monkeypatch) -> None:
    """A non-`list` reply yields an empty mapping, not a crash."""
    ws = _FakeWS(["welcome", json.dumps({"type": "something-else"})])
    _patch_connect(monkeypatch, ws)

    assert webrtc_utils.get_producer_list("host", 8443) == {}


def test_find_producer_peer_id_by_name(monkeypatch) -> None:
    """Returns the id of the first producer whose meta name matches."""
    reply = json.dumps(
        {
            "type": "list",
            "producers": [
                {"id": "peerA", "meta": {"name": "alice"}},
                {"id": "peerB", "meta": {"name": "bob"}},
            ],
        }
    )
    _patch_connect(monkeypatch, _FakeWS(["welcome", reply]))

    assert webrtc_utils.find_producer_peer_id_by_name("host", 8443, "bob") == "peerB"


def test_find_producer_peer_id_by_name_missing_raises(monkeypatch) -> None:
    """Missing name raises KeyError."""
    reply = json.dumps(
        {"type": "list", "producers": [{"id": "peerA", "meta": {"name": "alice"}}]}
    )
    _patch_connect(monkeypatch, _FakeWS(["welcome", reply]))

    with pytest.raises(KeyError):
        webrtc_utils.find_producer_peer_id_by_name("host", 8443, "nobody")
