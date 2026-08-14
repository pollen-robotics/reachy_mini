"""Tests for the WebRTC signalling helpers.

`connect` is mocked with a scripted fake websocket, so the producer-list
request/parse logic is covered without a real signalling server. The TURN
helpers are exercised without network or threads.
"""

import json
from pathlib import Path

import pytest

from reachy_mini.daemon import startup_app_config
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


# --- TURN helpers -----------------------------------------------------


def test_ice_servers_to_turn_uris_builds_userinfo_uris() -> None:
    """A turn entry becomes `turn://user:pass@host:port`; `urls` may be a str."""
    servers = [{"urls": "turn:relay.example:3478", "username": "u", "credential": "p"}]

    assert webrtc_utils.ice_servers_to_turn_uris(servers) == [
        "turn://u:p@relay.example:3478"
    ]


def test_ice_servers_to_turn_uris_expands_url_lists() -> None:
    """One entry with several urls yields one URI per url, turns included."""
    servers = [
        {
            "urls": ["turn:relay.example:3478", "turns:relay.example:5349"],
            "username": "u",
            "credential": "p",
        }
    ]

    assert webrtc_utils.ice_servers_to_turn_uris(servers) == [
        "turn://u:p@relay.example:3478",
        "turns://u:p@relay.example:5349",
    ]


def test_ice_servers_to_turn_uris_skips_stun_and_credentialless() -> None:
    """STUN goes through webrtcbin's own property; no creds means no URI."""
    servers = [
        {"urls": "stun:stun.example:3478"},
        {"urls": "stun:stun.example:3478", "username": "u", "credential": "p"},
        {"urls": "turn:relay.example:3478"},
        {"urls": "turn:relay.example:3478", "username": "u"},
        {"urls": []},
    ]

    assert webrtc_utils.ice_servers_to_turn_uris(servers) == []


def test_ice_servers_to_turn_uris_percent_encodes_credentials() -> None:
    """A credential containing `:@/` must not be able to corrupt the URI."""
    servers = [
        {
            "urls": "turn:relay.example:3478",
            "username": "user@example.com",
            "credential": "a:b@c/d",
        }
    ]

    (uri,) = webrtc_utils.ice_servers_to_turn_uris(servers)

    assert uri == "turn://user%40example.com:a%3Ab%40c%2Fd@relay.example:3478"
    # Exactly one `@`, so the host half is unambiguous.
    assert uri.count("@") == 1
    assert uri.rsplit("@", 1)[1] == "relay.example:3478"


def test_turn_credentials_uris_empty_before_any_fetch() -> None:
    """`turn_uris` is safe to call before start(): no creds, no blocking."""
    assert webrtc_utils.TurnCredentials().turn_uris() == []


def test_turn_credentials_refresh_once_populates_cache(monkeypatch) -> None:
    """A successful fetch is converted to URIs and cached for readers."""
    creds = webrtc_utils.TurnCredentials(
        url="https://turn.example/credentials", ttl=600
    )

    class _Resp:
        @staticmethod
        def raise_for_status() -> None:
            pass

        @staticmethod
        def json() -> dict:
            return {
                "iceServers": [
                    {"urls": "stun:stun.example:3478"},
                    {
                        "urls": "turn:relay.example:3478",
                        "username": "u",
                        "credential": "p",
                    },
                ]
            }

    seen: dict = {}

    def _fake_get(url, **kwargs):
        seen["url"] = url
        seen["headers"] = kwargs.get("headers")
        seen["params"] = kwargs.get("params")
        return _Resp()

    monkeypatch.setattr(webrtc_utils, "requests", type("R", (), {"get": _fake_get}))
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "hf_tok", raising=False)

    assert creds._refresh_once() == 300.0  # half the 600 s TTL
    assert creds.turn_uris() == ["turn://u:p@relay.example:3478"]
    assert seen["url"] == "https://turn.example/credentials"
    assert seen["headers"]["Authorization"] == "Bearer hf_tok"
    assert seen["params"] == {"ttl": 600}


def test_turn_credentials_no_url_start_is_a_noop(caplog) -> None:
    """Without a URL (the default) start() spawns nothing and says so once."""
    creds = webrtc_utils.TurnCredentials(url="")

    with caplog.at_level("INFO", logger=webrtc_utils.__name__):
        creds.start()

    assert creds._thread is None
    assert sum("TURN relay disabled" in r.message for r in caplog.records) == 1


def test_turn_credentials_network_failure_retries_soon(monkeypatch) -> None:
    """A transient failure backs off briefly and keeps the last good creds."""
    creds = webrtc_utils.TurnCredentials(url="https://turn.example/credentials")
    creds._uris = ["turn://u:p@relay.example:3478"]

    def _boom(*_args, **_kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(webrtc_utils, "requests", type("R", (), {"get": _boom}))
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "hf_tok", raising=False)

    assert creds._refresh_once() == webrtc_utils._TURN_RETRY_AFTER_FAILURE_S
    assert creds.turn_uris() == ["turn://u:p@relay.example:3478"]


def test_turn_credentials_without_token_does_not_fetch(monkeypatch) -> None:
    """No HF token means no request at all, and no crash."""
    creds = webrtc_utils.TurnCredentials()

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("must not request TURN creds without a token")

    monkeypatch.setattr(webrtc_utils, "requests", type("R", (), {"get": _unexpected}))
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None, raising=False)

    assert creds._refresh_once() > webrtc_utils._TURN_RETRY_AFTER_FAILURE_S
    assert creds.turn_uris() == []


def test_turn_credentials_without_token_logs_once(monkeypatch, caplog) -> None:
    """Not being logged in is a steady state, not a failure to retry loudly.

    This thread runs for the daemon's whole life, so a short backoff plus a
    per-attempt warning would put a line in the log forever.
    """
    creds = webrtc_utils.TurnCredentials(ttl=600)
    monkeypatch.setattr(
        webrtc_utils,
        "requests",
        type("R", (), {"get": lambda *a, **k: pytest.fail("must not fetch")}),
    )
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None, raising=False)

    with caplog.at_level("INFO", logger=webrtc_utils.__name__):
        delays = [creds._refresh_once() for _ in range(5)]

    assert delays == [300.0] * 5  # full period, not the failure backoff
    assert sum("No HF token" in r.message for r in caplog.records) == 1


def test_get_turn_enabled_reads_daemon_config(tmp_path, monkeypatch) -> None:
    """`turn_enabled` is read as a bool; unset or malformed means None."""
    cfg = Path(tmp_path) / "daemon_config.json"
    monkeypatch.setattr(startup_app_config, "_config_path", lambda: cfg)

    assert startup_app_config.get_turn_enabled() is None  # missing file

    cfg.write_text(json.dumps({"turn_enabled": False}))
    assert startup_app_config.get_turn_enabled() is False

    cfg.write_text(json.dumps({"turn_enabled": True}))
    assert startup_app_config.get_turn_enabled() is True

    # A string is not a bool: fall back rather than treat "false" as truthy.
    cfg.write_text(json.dumps({"turn_enabled": "false"}))
    assert startup_app_config.get_turn_enabled() is None

    # Unrelated keys are left alone.
    cfg.write_text(json.dumps({"startup_app": "some-app"}))
    assert startup_app_config.get_turn_enabled() is None
