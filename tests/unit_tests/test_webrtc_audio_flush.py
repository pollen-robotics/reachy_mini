"""Tests for WebRTC audio appsink flushing on stop/start recording."""

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")

from unittest.mock import MagicMock  # noqa: E402

from reachy_mini.media.webrtc_client_gstreamer import GstWebRTCClient  # noqa: E402


def _make_client():
    """Build a GstWebRTCClient instance without running its full __init__."""
    client = GstWebRTCClient.__new__(GstWebRTCClient)
    client.logger = MagicMock()
    client._appsink_audio = MagicMock()
    return client


def test_flush_drains_queued_samples():
    """_flush_audio_appsink pulls every queued sample until the queue is empty."""
    client = _make_client()
    samples = [object(), object(), object()]
    queue = list(samples)

    def try_pull_sample(timeout):
        if queue:
            return queue.pop(0)
        return None

    client._appsink_audio.try_pull_sample.side_effect = try_pull_sample
    client._flush_audio_appsink()
    assert queue == []
    # try_pull_sample called until None returned
    assert client._appsink_audio.try_pull_sample.call_count == len(samples) + 1


def test_flush_empty_queue_is_noop():
    """Flushing an empty queue does not raise and calls try_pull_sample once."""
    client = _make_client()
    client._appsink_audio.try_pull_sample.return_value = None
    client._flush_audio_appsink()
    assert client._appsink_audio.try_pull_sample.call_count == 1


def test_flush_handles_appsink_error():
    """An exception while flushing is logged and not re-raised."""
    client = _make_client()
    client._appsink_audio.try_pull_sample.side_effect = RuntimeError("boom")
    client._flush_audio_appsink()  # must not raise
    client.logger.debug.assert_called()


def test_start_and_stop_recording_flush():
    """start_recording and stop_recording both drain the appsink queue."""
    client = _make_client()
    calls = {"n": 0}

    def try_pull_sample(timeout):
        calls["n"] += 1
        return None

    client._appsink_audio.try_pull_sample.side_effect = try_pull_sample
    client.start_recording()
    client.stop_recording()
    # Each call drains (at least one try_pull_sample per flush).
    assert calls["n"] >= 2


def test_stop_resume_discards_stale_samples():
    """Samples queued while stopped are drained on stop and on resume."""
    client = _make_client()
    queue = [b"stale-1", b"stale-2"]

    def try_pull_sample(timeout):
        if queue:
            return queue.pop(0)
        return None

    client._appsink_audio.try_pull_sample.side_effect = try_pull_sample

    # While stopped, stale samples accumulate; stop_recording drains them.
    client.stop_recording()
    assert queue == [], "stop_recording did not drain stale samples"

    # Simulate samples arriving during the stopped window.
    queue.extend([b"stale-3", b"stale-4"])
    # Resuming drains everything queued so far.
    client.start_recording()
    assert queue == [], "start_recording did not drain stale samples"

    # After resume, the queue starts empty; nothing stale remains to leak.
    client._appsink_audio.try_pull_sample.side_effect = lambda t: None
    assert client._appsink_audio.try_pull_sample(0) is None
