"""Tests for dropping stale WebRTC audio on start_recording."""

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")

from unittest.mock import MagicMock  # noqa: E402

from reachy_mini.media.webrtc_client_gstreamer import GstWebRTCClient  # noqa: E402


def test_start_recording_drops_queued_samples():
    """Samples queued while stopped are gone after start_recording."""
    client = GstWebRTCClient.__new__(GstWebRTCClient)
    queue = [object(), object(), object()]
    client._appsink_audio = MagicMock()
    client._appsink_audio.try_pull_sample.side_effect = (
        lambda _: queue.pop(0) if queue else None
    )

    client.start_recording()

    assert queue == []
