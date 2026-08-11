"""Construct the daemon media server against the gst-plugins-rs webrtcsink.

`GstMediaServer.__init__` hard-requires the `webrtcsink` element, so the class
was untestable without ``libgstrswebrtc.so``. With the plugin loaded we can
construct it in simulation mode — no camera, no PulseAudio needed — which
exercises the whole pipeline-building path (webrtc/video-sim/audio/IPC).

Skipped where the plugin is absent.
"""

from __future__ import annotations

import pytest

from reachy_mini.media.camera_constants import MujocoCameraSpecs
from reachy_mini.media.media_server import GstMediaServer, SimulationMode

pytestmark = pytest.mark.webrtc


def test_media_server_constructs_in_mujoco_sim() -> None:
    """The MUJOCO sim path (UDP video, no camera) builds the full pipeline."""
    server = GstMediaServer(sim_mode=SimulationMode.MUJOCO)
    try:
        assert isinstance(server.camera_specs, MujocoCameraSpecs)
        assert server._resolution is not None
    finally:
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
