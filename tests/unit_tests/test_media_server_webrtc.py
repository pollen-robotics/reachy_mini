"""GStreamer media-server integration tests."""

from unittest.mock import MagicMock

import pytest

from reachy_mini.media import media_server
from reachy_mini.media.camera_constants import MujocoCameraSpecs
from reachy_mini.media.media_server import GstMediaServer, SimulationMode

pytestmark = pytest.mark.webrtc


def test_media_server_lifecycle_reuses_initial_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initial start reuses its pipeline and restart rebuilds it."""
    pipeline_new = MagicMock(wraps=media_server.Gst.Pipeline.new)
    monkeypatch.setattr(media_server.Gst.Pipeline, "new", pipeline_new)
    server = GstMediaServer(sim_mode=SimulationMode.MUJOCO)
    try:
        assert isinstance(server.camera_specs, MujocoCameraSpecs)
        assert pipeline_new.call_count == 1

        server.start()
        assert pipeline_new.call_count == 1

        server.stop()
        server.start()
        assert pipeline_new.call_count == 2
    finally:
        server.stop()
        server.close()


def test_enable_turn_false_starts_no_refresher() -> None:
    """Opting out means no credentials object, hence no thread and no fetch."""
    server = GstMediaServer(sim_mode=SimulationMode.MUJOCO, enable_turn=False)
    try:
        assert server._turn is None
        # The consumer hook stays callable; it just adds nothing.
        server._apply_turn_servers(None)  # type: ignore[arg-type]
    finally:
        server.close()


def test_rpi_encoder_probe_requires_runtime_bitrate_property(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simplified pipeline only when v4l2h264enc exposes a `bitrate` property."""
    from types import SimpleNamespace

    from gi.repository import Gst

    def fake_encoder(present: bool, has_bitrate: bool) -> None:
        enc = SimpleNamespace(
            find_property=lambda name: object()
            if has_bitrate and name == "bitrate"
            else None
        )
        monkeypatch.setattr(
            Gst.ElementFactory, "make", lambda name: enc if present else None
        )

    fake_encoder(present=False, has_bitrate=False)
    assert GstMediaServer._webrtcsink_handles_rpi_encoder() is False
    # Old OS image: stock v4l2h264enc, bitrate only via extra-controls
    fake_encoder(present=True, has_bitrate=False)
    assert GstMediaServer._webrtcsink_handles_rpi_encoder() is False
    # New OS image (reachy-mini-os#65): patched encoder with runtime bitrate
    fake_encoder(present=True, has_bitrate=True)
    assert GstMediaServer._webrtcsink_handles_rpi_encoder() is True
