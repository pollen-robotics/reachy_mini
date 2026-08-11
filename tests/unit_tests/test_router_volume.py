"""Unit tests for the volume router (speaker + microphone + test sound)."""

from unittest.mock import MagicMock

import pytest

from reachy_mini.daemon.app.routers import volume


class _Device:
    """Audio device stand-in exposing only a name."""

    def __init__(self, name: str) -> None:
        self.name = name


class _StubVolumeControl:
    """In-memory VolumeControl — no audio hardware, fully scriptable."""

    def __init__(self, output: int = 42, mic: int = 30, set_ok: bool = True) -> None:
        self._output = output
        self._mic = mic
        self._set_ok = set_ok
        self.platform_name = "TestOS"
        self.output_device = _Device("Speaker")
        self.input_device = _Device("Mic")

    def get_output_volume(self) -> int:
        """Return the scripted output volume."""
        return self._output

    def set_output_volume(self, volume: int) -> bool:
        """Record the write and report the scripted outcome."""
        self._output = volume
        return self._set_ok

    def get_input_volume(self) -> int:
        """Return the scripted input volume."""
        return self._mic

    def set_input_volume(self, volume: int) -> bool:
        """Record the write and report the scripted outcome."""
        self._mic = volume
        return self._set_ok


def _patch_vc(monkeypatch, stub: _StubVolumeControl) -> None:
    """Route the module accessor to the stub."""
    monkeypatch.setattr(volume, "_get_volume_control", lambda: stub)


def test_get_current_output_volume(monkeypatch, router_app):
    """GET /volume/current -> 200 with the stub's output value."""
    _patch_vc(monkeypatch, _StubVolumeControl(output=42))
    client = router_app(volume.router)

    resp = client.get("/volume/current")

    assert resp.status_code == 200
    assert resp.json() == {"volume": 42, "platform": "TestOS", "device": "Speaker"}


def test_get_microphone_volume(monkeypatch, router_app):
    """GET /volume/microphone/current -> 200 with the stub's input value."""
    _patch_vc(monkeypatch, _StubVolumeControl(mic=30))
    client = router_app(volume.router)

    resp = client.get("/volume/microphone/current")

    assert resp.status_code == 200
    assert resp.json() == {"volume": 30, "platform": "TestOS", "device": "Mic"}


def test_get_output_volume_failure(monkeypatch, router_app):
    """Negative read -> 500 failure detail."""
    _patch_vc(monkeypatch, _StubVolumeControl(output=-1))
    client = router_app(volume.router)

    resp = client.get("/volume/current")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to get volume"


def test_get_microphone_volume_failure(monkeypatch, router_app):
    """Negative mic read -> 500 failure detail."""
    _patch_vc(monkeypatch, _StubVolumeControl(mic=-1))
    client = router_app(volume.router)

    resp = client.get("/volume/microphone/current")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to get microphone volume"


def test_set_output_volume_success(monkeypatch, router_app):
    """POST /volume/set with a working setter and no daemon -> 200."""
    _patch_vc(monkeypatch, _StubVolumeControl(set_ok=True))
    client = router_app(volume.router)

    resp = client.post("/volume/set", json={"volume": 55})

    assert resp.status_code == 200
    assert resp.json() == {"volume": 55, "platform": "TestOS", "device": "Speaker"}


def test_set_output_volume_plays_test_sound(monkeypatch, router_app):
    """A ready backend on app.state triggers a test sound after the set."""
    _patch_vc(monkeypatch, _StubVolumeControl(set_ok=True))
    daemon = MagicMock()
    daemon.backend.ready.is_set.return_value = True
    client = router_app(volume.router, daemon=daemon)

    resp = client.post("/volume/set", json={"volume": 55})

    assert resp.status_code == 200
    daemon.backend.play_sound.assert_called_once_with("impatient1.wav")


def test_set_output_volume_test_sound_failure_ignored(monkeypatch, router_app):
    """A failing test sound is swallowed — the set still returns 200."""
    _patch_vc(monkeypatch, _StubVolumeControl(set_ok=True))
    daemon = MagicMock()
    daemon.backend.ready.is_set.return_value = True
    daemon.backend.play_sound.side_effect = RuntimeError("boom")
    client = router_app(volume.router, daemon=daemon)

    resp = client.post("/volume/set", json={"volume": 55})

    assert resp.status_code == 200


def test_set_output_volume_failure(monkeypatch, router_app):
    """Setter returning False -> 500."""
    _patch_vc(monkeypatch, _StubVolumeControl(set_ok=False))
    client = router_app(volume.router)

    resp = client.post("/volume/set", json={"volume": 55})

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to set volume"


def test_set_microphone_volume_success(monkeypatch, router_app):
    """POST /volume/microphone/set with a working setter -> 200."""
    _patch_vc(monkeypatch, _StubVolumeControl(set_ok=True))
    client = router_app(volume.router)

    resp = client.post("/volume/microphone/set", json={"volume": 20})

    assert resp.status_code == 200
    assert resp.json() == {"volume": 20, "platform": "TestOS", "device": "Mic"}


def test_set_microphone_volume_failure(monkeypatch, router_app):
    """Mic setter returning False -> 500."""
    _patch_vc(monkeypatch, _StubVolumeControl(set_ok=False))
    client = router_app(volume.router)

    resp = client.post("/volume/microphone/set", json={"volume": 20})

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to set microphone volume"


@pytest.mark.parametrize("route", ["/volume/set", "/volume/microphone/set"])
@pytest.mark.parametrize("bad", [-5, 150])
def test_set_volume_out_of_range_rejected(monkeypatch, router_app, route, bad):
    """Out-of-range volume is rejected by validation -> 422."""
    _patch_vc(monkeypatch, _StubVolumeControl())
    client = router_app(volume.router)

    resp = client.post(route, json={"volume": bad})

    assert resp.status_code == 422


def test_test_sound_success(router_app):
    """POST /volume/test-sound plays the sound -> 200 ok."""
    backend = MagicMock()
    client = router_app(volume.router, backend=backend)

    resp = client.post("/volume/test-sound")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "message": "Test sound played"}
    backend.play_sound.assert_called_once_with("impatient1.wav")


@pytest.mark.parametrize("msg", ["Device unavailable", "error -9985 occurred"])
def test_test_sound_device_busy(router_app, msg):
    """A busy audio device -> 200 busy, sound skipped."""
    backend = MagicMock()
    backend.play_sound.side_effect = RuntimeError(msg)
    client = router_app(volume.router, backend=backend)

    resp = client.post("/volume/test-sound")

    assert resp.status_code == 200
    assert resp.json()["status"] == "busy"


def test_test_sound_error(router_app):
    """An unexpected playback error -> 500."""
    backend = MagicMock()
    backend.play_sound.side_effect = RuntimeError("boom")
    client = router_app(volume.router, backend=backend)

    resp = client.post("/volume/test-sound")

    assert resp.status_code == 500
    assert "Failed to play test sound" in resp.json()["detail"]
