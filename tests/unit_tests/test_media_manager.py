"""Tests for reachy_mini.media.media_manager."""

from __future__ import annotations

import sys
import types

from reachy_mini.media.media_manager import MediaBackend, MediaManager


def test_webrtc_with_direct_camera_starts_both_paths(monkeypatch) -> None:
    """Start the WebRTC audio path even when direct camera fallback is selected."""

    camera_events: list[str] = []
    webrtc_events: list[str] = []

    class DummyCamera:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            camera_events.append("camera.init")

        def open(self) -> None:
            camera_events.append("camera.open")

        def close(self) -> None:
            camera_events.append("camera.close")

    class DummyWebRTC:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self.daemon_url = ""
            webrtc_events.append("webrtc.init")

        def open(self) -> None:
            webrtc_events.append("webrtc.open")

        def close(self) -> None:
            webrtc_events.append("webrtc.close")

        def stop_recording(self) -> None:
            webrtc_events.append("webrtc.stop_recording")

        def stop_playing(self) -> None:
            webrtc_events.append("webrtc.stop_playing")

        def cleanup(self) -> None:
            webrtc_events.append("webrtc.cleanup")

    monkeypatch.setitem(
        sys.modules,
        "reachy_mini.media.camera_opencv",
        types.SimpleNamespace(OpenCVCamera=DummyCamera),
    )
    monkeypatch.setitem(
        sys.modules,
        "reachy_mini.media.webrtc_client_gstreamer",
        types.SimpleNamespace(GstWebRTCClient=DummyWebRTC),
    )
    monkeypatch.setitem(
        sys.modules,
        "reachy_mini.media.webrtc_utils",
        types.SimpleNamespace(find_producer_peer_id_by_name=lambda *args: "peer-id"),
    )

    manager = MediaManager(
        backend=MediaBackend.WEBRTC,
        direct_camera_index=1,
    )

    assert camera_events == ["camera.init", "camera.open"]
    assert webrtc_events == ["webrtc.init", "webrtc.open"]

    manager.close()

    assert camera_events == ["camera.init", "camera.open", "camera.close"]
    assert webrtc_events == [
        "webrtc.init",
        "webrtc.open",
        "webrtc.stop_recording",
        "webrtc.stop_playing",
        "webrtc.close",
        "webrtc.cleanup",
    ]


def test_webrtc_with_named_direct_camera_prefers_ffmpeg_fallback(monkeypatch) -> None:
    """Prefer the FFmpeg direct camera backend when a camera name is provided."""

    camera_events: list[str] = []
    webrtc_events: list[str] = []

    class DummyFFmpegCamera:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            camera_events.append("camera.init")

        def open(self) -> None:
            camera_events.append("camera.open")

        def close(self) -> None:
            camera_events.append("camera.close")

    class DummyWebRTC:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self.daemon_url = ""
            webrtc_events.append("webrtc.init")

        def open(self) -> None:
            webrtc_events.append("webrtc.open")

        def close(self) -> None:
            webrtc_events.append("webrtc.close")

        def stop_recording(self) -> None:
            webrtc_events.append("webrtc.stop_recording")

        def stop_playing(self) -> None:
            webrtc_events.append("webrtc.stop_playing")

        def cleanup(self) -> None:
            webrtc_events.append("webrtc.cleanup")

    monkeypatch.setitem(
        sys.modules,
        "reachy_mini.media.camera_ffmpeg",
        types.SimpleNamespace(FFmpegCamera=DummyFFmpegCamera),
    )
    monkeypatch.setitem(
        sys.modules,
        "reachy_mini.media.webrtc_client_gstreamer",
        types.SimpleNamespace(GstWebRTCClient=DummyWebRTC),
    )
    monkeypatch.setitem(
        sys.modules,
        "reachy_mini.media.webrtc_utils",
        types.SimpleNamespace(find_producer_peer_id_by_name=lambda *args: "peer-id"),
    )

    manager = MediaManager(
        backend=MediaBackend.WEBRTC,
        direct_camera_name="Reachy Mini Camera",
    )

    assert camera_events == ["camera.init", "camera.open"]
    assert webrtc_events == ["webrtc.init", "webrtc.open"]

    manager.close()

    assert camera_events == ["camera.init", "camera.open", "camera.close"]
