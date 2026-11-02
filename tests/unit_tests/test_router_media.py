"""Tests for the media router."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

from reachy_mini.daemon.app.routers import media
from reachy_mini.media.media_server import GstMediaServer


def _daemon(play_sound: Mock) -> SimpleNamespace:
    ready = threading.Event()
    ready.set()
    return SimpleNamespace(backend=SimpleNamespace(ready=ready, play_sound=play_sound))


def test_play_sound_unknown_file_returns_404(router_app):
    """A file that is not uploaded, not a path and not an asset is a 404."""
    client = router_app(media.router, daemon=_daemon(Mock(return_value=False)))

    r = client.post("/media/play_sound", json={"file": "does_not_exist.wav"})

    assert r.status_code == 404
    assert "no such sound" in r.json()["detail"]


def test_play_sound_known_file_returns_ok(router_app):
    """A sound the backend starts is acknowledged."""
    client = router_app(media.router, daemon=_daemon(Mock(return_value=True)))

    r = client.post("/media/play_sound", json={"file": "wake_up.wav"})

    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_media_server_play_sound_reports_missing_file():
    """The media server says it played nothing for an unknown file."""
    server = SimpleNamespace(_logger=Mock())

    assert GstMediaServer.play_sound(server, "does_not_exist.wav") is False  # type: ignore[arg-type]
