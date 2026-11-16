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


def test_first_start_reuses_the_pipeline_built_in_init() -> None:
    """Only a start() after stop() rebuilds the pipeline."""
    server = GstMediaServer(sim_mode=SimulationMode.MUJOCO, enable_turn=False)
    built = server._pipeline_sender
    bus = server._bus_sender
    try:
        server.start()
        assert server._pipeline_sender is built

        server.stop()
        server.start()
        assert server._pipeline_sender is not built
        assert bus.remove_watch() is False  # already dropped by the rebuild
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


def test_disable_upnp_turns_off_the_nice_agent_port_mapping() -> None:
    """Consumers never ask the home router for UPnP port mappings."""
    import gc
    import logging
    import types

    from gi.repository import Gst

    Gst.init(None)
    webrtcbin = Gst.ElementFactory.make("webrtcbin")
    if webrtcbin is None:
        pytest.skip("webrtcbin not available")
    ice = webrtcbin.get_property("ice-agent")
    from reachy_mini.media.media_server import _keep_ice_agent_alive

    _keep_ice_agent_alive(ice)  # reading the property alone would free it
    agent = ice.get_property("agent")
    assert agent.get_property("upnp") is True  # libnice default
    del ice

    del agent
    fake = types.SimpleNamespace(_logger=logging.getLogger("test"))
    GstMediaServer._disable_upnp(fake, webrtcbin)  # type: ignore[arg-type]
    gc.collect()

    # The ICE agent survives the Python wrappers being collected (webrtcbin
    # hands out a floating reference) and keeps UPnP off.
    ice = webrtcbin.get_property("ice-agent")
    assert ice is not None
    assert ice.get_property("agent").get_property("upnp") is False


def test_disable_upnp_tolerates_an_unreachable_agent() -> None:
    """An ICE agent without a NiceAgent (older GStreamer) only logs."""
    import logging
    import types

    fake = types.SimpleNamespace(_logger=logging.getLogger("test"))
    GstMediaServer._disable_upnp(fake, object())  # type: ignore[arg-type]
