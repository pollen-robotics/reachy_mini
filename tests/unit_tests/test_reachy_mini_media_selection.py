"""Tests for ReachyMini media backend selection."""

from __future__ import annotations

from types import SimpleNamespace

from reachy_mini.media.camera_constants import ReachyMiniLiteCamSpecs
from reachy_mini.media.media_manager import MediaBackend
from reachy_mini.reachy_mini import ReachyMini


def test_local_macos_uses_direct_camera_fallback_when_no_ipc(monkeypatch) -> None:
    """Prefer the direct macOS camera fallback for local apps when IPC is absent."""

    captured_args: dict[str, object] = {}

    class DummyClient:
        host = "localhost"
        port = 8000

        def get_status(self) -> SimpleNamespace:
            return SimpleNamespace(
                no_media=False,
                camera_specs_name="lite",
                wlan_ip=None,
            )

        def disconnect(self) -> None:
            pass

    class DummyMediaManager:
        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            captured_args.update(kwargs)

        def close(self) -> None:
            pass

    monkeypatch.setattr("reachy_mini.reachy_mini.daemon_check", lambda *args: None)
    monkeypatch.setattr(
        ReachyMini,
        "_initialize_client",
        lambda self, requested_mode, timeout: (DummyClient(), "localhost_only"),
    )
    monkeypatch.setattr(
        ReachyMini,
        "set_automatic_body_yaw",
        lambda self, automatic_body_yaw: None,
    )
    monkeypatch.setattr(
        "reachy_mini.reachy_mini.is_local_camera_available",
        lambda: False,
    )
    monkeypatch.setattr(
        "reachy_mini.reachy_mini.get_macos_ffmpeg_video_device",
        lambda: ("1", ReachyMiniLiteCamSpecs()),
    )
    monkeypatch.setattr("reachy_mini.reachy_mini.MediaManager", DummyMediaManager)

    mini = ReachyMini()

    assert captured_args["backend"] == MediaBackend.WEBRTC
    assert captured_args["direct_camera_index"] == 1
    assert isinstance(captured_args["camera_specs"], ReachyMiniLiteCamSpecs)

    mini.client.disconnect()
