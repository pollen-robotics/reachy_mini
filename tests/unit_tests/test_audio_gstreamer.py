"""Unit tests for the shared appsrc PTS helper and the SW AEC detection."""

import logging
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from reachy_mini.media.audio_aec import resolve_sw_aec_enabled
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


# --- Software AEC auto-detection ---------------------------------------
#
# These tests cover the shared helper in `reachy_mini.media.audio_aec`,
# which is consumed by both `GStreamerAudio` (LOCAL Python SDK audio
# backend) and `GstMediaServer` (WebRTC daemon pipeline). The helper is
# a pure function so we hit it directly instead of going through one of
# the two backends.


def _silent_logger() -> logging.Logger:
    """Return a logger that swallows everything during the unit test."""
    logger = logging.getLogger("reachy_mini.test.audio_aec")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL + 1)
    return logger


def test_auto_disabled_when_asoundrc_present() -> None:
    """Wireless `.asoundrc` always wins (XMOS loopback handles AEC)."""
    with patch(
        "reachy_mini.media.audio_aec.has_reachymini_asoundrc",
        return_value=True,
    ), patch(
        "reachy_mini.media.audio_aec.get_audio_device"
    ) as get_dev:
        result = resolve_sw_aec_enabled(_silent_logger())

    assert result is False
    # No reason to keep poking the device monitor once we know XMOS
    # is in the path.
    get_dev.assert_not_called()


def test_auto_disabled_when_respeaker_card_detected() -> None:
    """ReSpeaker USB dongle present (Lite happy path) skips SW AEC."""
    with patch(
        "reachy_mini.media.audio_aec.has_reachymini_asoundrc",
        return_value=False,
    ), patch(
        "reachy_mini.media.audio_aec.get_audio_device",
        return_value="reachy-mini-audio-id",
    ):
        result = resolve_sw_aec_enabled(_silent_logger())

    assert result is False


def test_auto_enabled_when_no_hardware_and_plugins_present() -> None:
    """Sim / dev box: no hardware AEC, plugins available -> enable SW AEC."""
    with patch(
        "reachy_mini.media.audio_aec.has_reachymini_asoundrc",
        return_value=False,
    ), patch(
        "reachy_mini.media.audio_aec.get_audio_device",
        return_value=None,
    ), patch(
        "reachy_mini.media.audio_aec.Gst.ElementFactory.find",
        return_value=object(),
    ):
        result = resolve_sw_aec_enabled(_silent_logger())

    assert result is True


def test_auto_disabled_when_no_hardware_but_plugins_missing() -> None:
    """No hardware AND no `webrtcdsp` packaged -> fall back to no AEC (warn only)."""
    with patch(
        "reachy_mini.media.audio_aec.has_reachymini_asoundrc",
        return_value=False,
    ), patch(
        "reachy_mini.media.audio_aec.get_audio_device",
        return_value=None,
    ), patch(
        "reachy_mini.media.audio_aec.Gst.ElementFactory.find",
        return_value=None,
    ):
        result = resolve_sw_aec_enabled(_silent_logger())

    assert result is False
