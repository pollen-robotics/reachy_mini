"""Tests for reachy_mini.media.media_server."""

from __future__ import annotations

from reachy_mini.daemon.utils import SimulationMode
from reachy_mini.media.camera_constants import ReachyMiniLiteCamSpecs
from reachy_mini.media.media_server import GstMediaServer


def test_media_server_detects_devices_before_starting_glib_loop(
    monkeypatch,
) -> None:
    """Run device discovery before the GLib loop thread starts.

    macOS can segfault in ``Gst.DeviceMonitor`` when the background GLib loop
    is already running. Keep the constructor ordering stable so discovery stays
    single-threaded.
    """
    events: list[str] = []

    class DummyLoop:
        def run(self) -> None:
            events.append("loop.run")

        def quit(self) -> None:
            events.append("loop.quit")

    class DummyThread:
        def __init__(self, target, daemon: bool) -> None:  # type: ignore[no-untyped-def]
            self._target = target
            self._started = False

        def start(self) -> None:
            events.append("thread.start")
            self._started = True

        def is_alive(self) -> bool:
            return self._started

    class DummyBus:
        def remove_watch(self) -> None:
            events.append("bus.remove_watch")

    def fake_get_video_device():
        events.append("get_video_device")
        assert "thread.start" not in events
        return "", None

    def fake_build_pipeline(self: GstMediaServer) -> None:
        events.append("build_pipeline")
        self._bus_sender = DummyBus()

    monkeypatch.setattr("reachy_mini.media.media_server.Gst.init", lambda _args: None)
    monkeypatch.setattr("reachy_mini.media.media_server.GLib.MainLoop", DummyLoop)
    monkeypatch.setattr("reachy_mini.media.media_server.Thread", DummyThread)
    monkeypatch.setattr(
        "reachy_mini.media.media_server.get_video_device", fake_get_video_device
    )
    monkeypatch.setattr(GstMediaServer, "_build_pipeline", fake_build_pipeline)

    server = GstMediaServer(log_level="INFO", sim_mode=SimulationMode.NONE)

    assert events == [
        "get_video_device",
        "build_pipeline",
        "thread.start",
    ]
    assert isinstance(server.camera_specs, ReachyMiniLiteCamSpecs)

    server.close()
    assert events[-2:] == ["loop.quit", "bus.remove_watch"]
