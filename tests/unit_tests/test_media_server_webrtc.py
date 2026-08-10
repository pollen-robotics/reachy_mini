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
