"""Unit tests for GStreamer audio playback timestamp helpers."""

from typing import cast

from reachy_mini.media.audio_gstreamer import GStreamerAudio


def test_compute_playback_buffer_timing_starts_at_running_time() -> None:
    """Start the first buffer at the current playback running time."""
    pts_ns, duration_ns, next_pts_ns = GStreamerAudio._compute_playback_buffer_timing(
        cast(GStreamerAudio, object()),
        1600,
        16000,
        2_000_000_000,
        None,
        GStreamerAudio.PLAYBACK_GAP_RESET_NS,
    )

    assert pts_ns == 2_000_000_000
    assert duration_ns == 100_000_000
    assert next_pts_ns == 2_100_000_000


def test_compute_playback_buffer_timing_continues_without_gap() -> None:
    """Keep appending buffers when the running time has not drifted ahead."""
    pts_ns, duration_ns, next_pts_ns = GStreamerAudio._compute_playback_buffer_timing(
        cast(GStreamerAudio, object()),
        800,
        16000,
        1_050_000_000,
        1_100_000_000,
        GStreamerAudio.PLAYBACK_GAP_RESET_NS,
    )

    assert pts_ns == 1_100_000_000
    assert duration_ns == 50_000_000
    assert next_pts_ns == 1_150_000_000


def test_compute_playback_buffer_timing_resets_after_large_gap() -> None:
    """Realign buffer timing after a long idle gap in sparse realtime audio."""
    pts_ns, duration_ns, next_pts_ns = GStreamerAudio._compute_playback_buffer_timing(
        cast(GStreamerAudio, object()),
        800,
        16000,
        1_400_000_000,
        1_100_000_000,
        GStreamerAudio.PLAYBACK_GAP_RESET_NS,
    )

    assert pts_ns == 1_400_000_000
    assert duration_ns == 50_000_000
    assert next_pts_ns == 1_450_000_000
