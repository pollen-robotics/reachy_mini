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


def _has(server: GstMediaServer, element_name: str) -> bool:
    """Whether the sender pipeline currently holds a named element."""
    return server._pipeline_sender.get_by_name(element_name) is not None


def test_disable_camera_builds_an_audio_only_pipeline() -> None:
    """Dropping the camera removes both video branches and keeps audio."""
    server = GstMediaServer(sim_mode=SimulationMode.MUJOCO)
    try:
        assert server.camera_enabled
        assert _has(server, "queue_webrtc")
        assert _has(server, "queue_ipc")
        had_audio = _has(server, "queue_audiosrc")

        server.disable_camera()

        assert not server.camera_enabled
        assert not _has(server, "queue_webrtc")
        assert not _has(server, "queue_ipc")
        # Audio is untouched: present afterwards exactly when it was before
        # (a machine with no audio device has no audio branch to begin with).
        assert _has(server, "queue_audiosrc") == had_audio
    finally:
        server.close()


def test_enable_camera_restores_the_video_branches() -> None:
    """Re-enabling rebuilds the pipeline with video back in place."""
    server = GstMediaServer(sim_mode=SimulationMode.MUJOCO)
    try:
        server.disable_camera()
        server.enable_camera()

        assert server.camera_enabled
        assert _has(server, "queue_webrtc")
        assert _has(server, "queue_ipc")
    finally:
        server.close()


def test_camera_toggles_are_idempotent() -> None:
    """Repeating a toggle is a no-op rather than a second rebuild."""
    server = GstMediaServer(sim_mode=SimulationMode.MUJOCO)
    try:
        server.disable_camera()
        server.disable_camera()
        assert not server.camera_enabled

        server.enable_camera()
        server.enable_camera()
        assert server.camera_enabled
    finally:
        server.close()


def test_toggling_the_camera_leaves_a_stopped_pipeline_stopped() -> None:
    """A toggle must not start a pipeline that was never started.

    `release_media()` stops the pipeline; a camera toggle arriving while media
    is released has to stay a flag change, or the daemon would silently
    re-acquire the hardware it just handed over.
    """
    server = GstMediaServer(sim_mode=SimulationMode.MUJOCO)
    try:
        assert not server._pipeline_running

        server.disable_camera()

        assert not server.camera_enabled
        assert not server._pipeline_running
    finally:
        server.close()
