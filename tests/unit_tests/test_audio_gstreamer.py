"""Unit tests for the shared appsrc PTS helper."""

from types import SimpleNamespace
from typing import cast

from reachy_mini.media.audio_base import AudioBase
from reachy_mini.media.audio_gstreamer import GStreamerAudio


def _fake_self() -> AudioBase:
    """Return a stand-in with just the constants ``_compute_pts`` reads."""
    return cast(
        AudioBase,
        SimpleNamespace(
            SAMPLE_RATE=GStreamerAudio.SAMPLE_RATE,
            GAP_RESET_NS=GStreamerAudio.GAP_RESET_NS,
        ),
    )


def test_compute_pts_starts_at_running_time() -> None:
    """Start the first buffer at the current playback running time."""
    pts_ns, duration_ns, next_pts_ns = GStreamerAudio._compute_pts(
        _fake_self(),
        1600,
        2_000_000_000,
        -1,
    )

    assert pts_ns == 2_000_000_000
    assert duration_ns == 100_000_000
    assert next_pts_ns == 2_100_000_000


def test_compute_pts_continues_without_gap() -> None:
    """Keep appending buffers when the running time has not drifted ahead."""
    pts_ns, duration_ns, next_pts_ns = GStreamerAudio._compute_pts(
        _fake_self(),
        800,
        1_050_000_000,
        1_100_000_000,
    )

    assert pts_ns == 1_100_000_000
    assert duration_ns == 50_000_000
    assert next_pts_ns == 1_150_000_000


def test_compute_pts_resets_after_large_gap() -> None:
    """Realign buffer timing after a long idle gap in sparse realtime audio."""
    pts_ns, duration_ns, next_pts_ns = GStreamerAudio._compute_pts(
        _fake_self(),
        800,
        1_400_000_000,
        1_100_000_000,
    )

    assert pts_ns == 1_400_000_000
    assert duration_ns == 50_000_000
    assert next_pts_ns == 1_450_000_000
