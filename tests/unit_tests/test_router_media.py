"""Tests for the media router."""

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

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


def test_play_sound_playback_failure_returns_500(router_app):
    """A file the media server finds but cannot play is a 500, not a 404."""
    play_sound = Mock(side_effect=RuntimeError("Failed to create playbin element"))
    client = router_app(media.router, daemon=_daemon(play_sound))

    r = client.post("/media/play_sound", json={"file": "wake_up.wav"})

    assert r.status_code == 500
    assert r.json()["detail"] == "Failed to create playbin element"


def test_acquire_failure_returns_500(router_app):
    """A pipeline that does not come back is a 500, not ok."""
    daemon = SimpleNamespace(
        acquire_media=AsyncMock(side_effect=RuntimeError("Media pipeline failed"))
    )
    client = router_app(media.router, daemon=daemon)

    r = client.post("/media/acquire")

    assert r.status_code == 500
    assert r.json()["detail"] == "Media pipeline failed"


# "" resolves to the assets directory itself.
@pytest.mark.parametrize("sound_file", ["does_not_exist.wav", ""])
def test_media_server_play_sound_reports_missing_file(sound_file):
    """The media server says it played nothing for anything but a file."""
    server = SimpleNamespace(_logger=Mock())

    assert GstMediaServer.play_sound(server, sound_file) is False  # type: ignore[arg-type]
