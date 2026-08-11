"""Tests for the sound-upload REST route (GHSA-m2pc-3q4q-w6jr).

The ``/media/sounds/upload`` endpoint must only accept genuine audio files:
the extension is allow-listed, the body size is capped, and the content is
validated with GStreamer before anything is written to disk. Cases that do not
need GStreamer run unconditionally; the content-validation cases are guarded by
``pytest.importorskip`` so they skip on a gst-less runner.
"""

import wave
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini.daemon.app.middleware import MaxBodySizeMiddleware
from reachy_mini.daemon.app.routers import media


@pytest.fixture
def sounds_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the upload directory at a throwaway temp location."""
    target = tmp_path / "sounds"
    monkeypatch.setattr(media, "SOUNDS_TMP_DIR", str(target))
    return target


@pytest.fixture
def client() -> TestClient:
    """Bare FastAPI test client with just the media router mounted."""
    app = FastAPI()
    app.include_router(media.router)
    return TestClient(app)


def _wav_bytes() -> bytes:
    """Return a minimal but genuine mono PCM WAV (0.1 s of silence)."""
    import io

    raw = io.BytesIO()
    with wave.open(raw, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 800)
    return raw.getvalue()


def _files_in(directory: Path) -> list[str]:
    """List regular files in *directory* (empty if it does not exist)."""
    if not directory.is_dir():
        return []
    return [p.name for p in directory.iterdir() if p.is_file()]


# --- dependency-free checks (extension / size / traversal) -----------------


def test_disallowed_extension_is_rejected(client: TestClient, sounds_dir: Path) -> None:
    """The advisory PoC: a .sh script must be refused and never stored."""
    resp = client.post(
        "/media/sounds/upload",
        files={"file": ("script.sh", b"#!/bin/sh\nrm -rf /\n", "text/plain")},
    )
    assert resp.status_code == 400, resp.text
    assert _files_in(sounds_dir) == []


def test_missing_extension_is_rejected(client: TestClient, sounds_dir: Path) -> None:
    """A traversal-style name resolving to an extension-less file is refused."""
    resp = client.post(
        "/media/sounds/upload",
        files={
            "file": ("/etc/passwd", b"root:x:0:0:root:/root:/bin/sh\n", "text/plain")
        },
    )
    assert resp.status_code == 400, resp.text
    assert _files_in(sounds_dir) == []


def test_dotdot_filename_is_rejected(client: TestClient, sounds_dir: Path) -> None:
    """A bare '..' filename is refused outright."""
    resp = client.post(
        "/media/sounds/upload",
        files={"file": ("..", b"whatever", "application/octet-stream")},
    )
    assert resp.status_code == 400, resp.text
    assert _files_in(sounds_dir) == []


def test_oversized_upload_is_rejected_by_middleware(sounds_dir: Path) -> None:
    """The middleware rejects an oversized Content-Length before the body is read."""
    app = FastAPI()
    app.add_middleware(
        MaxBodySizeMiddleware,
        max_body_size=16,
        paths={"/media/sounds/upload"},
    )
    app.include_router(media.router)
    client = TestClient(app)

    resp = client.post(
        "/media/sounds/upload",
        files={"file": ("big.wav", b"\x00" * 1024, "audio/wav")},
    )
    assert resp.status_code == 413, resp.text
    assert _files_in(sounds_dir) == []


# --- content validation (requires GStreamer) -------------------------------


@pytest.fixture
def _require_gst() -> None:
    """Skip the test unless GStreamer pbutils is importable."""
    gi = pytest.importorskip("gi")
    gi.require_version("Gst", "1.0")
    gi.require_version("GstPbutils", "1.0")
    pytest.importorskip("gi.repository.GstPbutils")


@pytest.mark.usefixtures("_require_gst")
def test_valid_wav_is_stored(client: TestClient, sounds_dir: Path) -> None:
    """A genuine WAV is accepted and written under the sounds directory."""
    payload = _wav_bytes()
    resp = client.post(
        "/media/sounds/upload",
        files={"file": ("hello.wav", payload, "audio/wav")},
    )
    assert resp.status_code == 200, resp.text
    stored = Path(resp.json()["path"])
    assert stored == sounds_dir / "hello.wav"
    assert stored.read_bytes() == payload
    assert _files_in(sounds_dir) == ["hello.wav"]


@pytest.mark.usefixtures("_require_gst")
def test_non_audio_content_with_audio_extension_is_rejected(
    client: TestClient, sounds_dir: Path
) -> None:
    """A script disguised with a .wav extension fails content validation."""
    resp = client.post(
        "/media/sounds/upload",
        files={"file": ("evil.wav", b"#!/bin/sh\nrm -rf /\n", "audio/wav")},
    )
    assert resp.status_code == 400, resp.text
    assert _files_in(sounds_dir) == []


@pytest.mark.usefixtures("_require_gst")
def test_traversal_filename_is_stored_as_basename(
    client: TestClient, sounds_dir: Path
) -> None:
    """A traversal prefix is stripped; the file stays inside the sounds dir."""
    resp = client.post(
        "/media/sounds/upload",
        files={"file": ("../../../tmp/evil.wav", _wav_bytes(), "audio/wav")},
    )
    assert resp.status_code == 200, resp.text
    assert Path(resp.json()["path"]) == sounds_dir / "evil.wav"
    assert _files_in(sounds_dir) == ["evil.wav"]
