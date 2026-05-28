"""Unit tests for the shared appsrc PTS helper."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

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


# ─── Software AEC auto-detection ─────────────────────────────────────────


def _fake_aec_self() -> GStreamerAudio:
    """Return a stand-in with just what `_resolve_sw_aec_enabled` reads.

    The method only needs `self.logger`, so a `SimpleNamespace`
    with a no-op logger is sufficient — no need to spin up the
    GStreamer pipelines.
    """

    class _DummyLogger:
        def info(self, *_a, **_kw) -> None:  # noqa: D401
            """Silence info logs in tests."""

        def warning(self, *_a, **_kw) -> None:  # noqa: D401
            """Silence warnings in tests."""

    return cast(GStreamerAudio, SimpleNamespace(logger=_DummyLogger()))


def test_auto_disabled_when_asoundrc_present() -> None:
    """Wireless `.asoundrc` always wins (XMOS loopback handles AEC)."""
    with patch(
        "reachy_mini.media.audio_gstreamer.has_reachymini_asoundrc",
        return_value=True,
    ), patch(
        "reachy_mini.media.audio_gstreamer.get_audio_device"
    ) as get_dev:
        result = GStreamerAudio._resolve_sw_aec_enabled(_fake_aec_self())

    assert result is False
    # No reason to keep poking the device monitor once we know XMOS
    # is in the path.
    get_dev.assert_not_called()


def test_auto_disabled_when_respeaker_card_detected() -> None:
    """ReSpeaker USB dongle present (Lite happy path) skips SW AEC."""
    with patch(
        "reachy_mini.media.audio_gstreamer.has_reachymini_asoundrc",
        return_value=False,
    ), patch(
        "reachy_mini.media.audio_gstreamer.get_audio_device",
        return_value="reachy-mini-audio-id",
    ):
        result = GStreamerAudio._resolve_sw_aec_enabled(_fake_aec_self())

    assert result is False


def test_auto_enabled_when_no_hardware_and_plugins_present() -> None:
    """Sim / dev box: no hardware AEC, plugins available → enable SW AEC."""
    with patch(
        "reachy_mini.media.audio_gstreamer.has_reachymini_asoundrc",
        return_value=False,
    ), patch(
        "reachy_mini.media.audio_gstreamer.get_audio_device",
        return_value=None,
    ), patch(
        "reachy_mini.media.audio_gstreamer.Gst.ElementFactory.find",
        return_value=object(),
    ):
        result = GStreamerAudio._resolve_sw_aec_enabled(_fake_aec_self())

    assert result is True


def test_auto_disabled_when_no_hardware_but_plugins_missing() -> None:
    """No hardware AND no `webrtcdsp` packaged → fall back to no AEC (warn only)."""
    with patch(
        "reachy_mini.media.audio_gstreamer.has_reachymini_asoundrc",
        return_value=False,
    ), patch(
        "reachy_mini.media.audio_gstreamer.get_audio_device",
        return_value=None,
    ), patch(
        "reachy_mini.media.audio_gstreamer.Gst.ElementFactory.find",
        return_value=None,
    ):
        result = GStreamerAudio._resolve_sw_aec_enabled(_fake_aec_self())

    assert result is False
