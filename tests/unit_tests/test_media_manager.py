"""Tests for reachy_mini.media.media_manager."""

from __future__ import annotations

import sys
import types

from reachy_mini.media.media_manager import MediaBackend, MediaManager


def test_webrtc_initializes_single_media_path(monkeypatch) -> None:
    """WEBRTC should use a single GstWebRTCClient for both audio and video."""

    webrtc_events: list[str] = []

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
        "reachy_mini.media.webrtc_client_gstreamer",
        types.SimpleNamespace(GstWebRTCClient=DummyWebRTC),
    )
    monkeypatch.setitem(
        sys.modules,
        "reachy_mini.media.webrtc_utils",
        types.SimpleNamespace(find_producer_peer_id_by_name=lambda *args: "peer-id"),
    )

    manager = MediaManager(backend=MediaBackend.WEBRTC)

    assert manager.camera is manager.audio
    assert webrtc_events == ["webrtc.init", "webrtc.open"]

    manager.close()

    assert webrtc_events == [
        "webrtc.init",
        "webrtc.open",
        "webrtc.close",
        "webrtc.stop_recording",
        "webrtc.stop_playing",
        "webrtc.cleanup",
    ]
