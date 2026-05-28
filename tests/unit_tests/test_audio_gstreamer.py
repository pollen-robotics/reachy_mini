"""Unit tests for the shared appsrc PTS helper."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

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


# ─── Software AEC env-var override ───────────────────────────────────────


def _fake_aec_self(logger=None) -> GStreamerAudio:
    """Return a stand-in with just what `_resolve_sw_aec_enabled` reads.

    We avoid touching the GStreamer pipelines entirely; the method only
    needs `self.logger`, so a `SimpleNamespace` is sufficient.
    """

    class _DummyLogger:
        def info(self, *_a, **_kw) -> None:  # noqa: D401
            """Silence info logs in tests."""

        def warning(self, *_a, **_kw) -> None:  # noqa: D401
            """Silence warnings in tests."""

    return cast(
        GStreamerAudio,
        SimpleNamespace(logger=logger or _DummyLogger()),
    )


@pytest.mark.parametrize(
    "env_value",
    ["1", "true", "TRUE", "yes", "on"],
)
def test_disable_env_overrides_to_false(monkeypatch, env_value: str) -> None:
    """`REACHY_MINI_DISABLE_SW_AEC` short-circuits to False without probing."""
    monkeypatch.setenv("REACHY_MINI_DISABLE_SW_AEC", env_value)
    monkeypatch.delenv("REACHY_MINI_FORCE_SW_AEC", raising=False)

    # Detection helpers must not be called when env-var disables AEC.
    with patch(
        "reachy_mini.media.audio_gstreamer.has_reachymini_asoundrc"
    ) as soundrc, patch(
        "reachy_mini.media.audio_gstreamer.get_audio_device"
    ) as get_dev, patch(
        "reachy_mini.media.audio_gstreamer.Gst.ElementFactory.find"
    ) as find:
        result = GStreamerAudio._resolve_sw_aec_enabled(_fake_aec_self())

    assert result is False
    soundrc.assert_not_called()
    get_dev.assert_not_called()
    find.assert_not_called()


def test_force_env_overrides_to_true_when_plugins_present(monkeypatch) -> None:
    """`REACHY_MINI_FORCE_SW_AEC` enables AEC even with hardware detected."""
    monkeypatch.delenv("REACHY_MINI_DISABLE_SW_AEC", raising=False)
    monkeypatch.setenv("REACHY_MINI_FORCE_SW_AEC", "1")

    with patch(
        "reachy_mini.media.audio_gstreamer.has_reachymini_asoundrc",
        return_value=True,
    ) as soundrc, patch(
        "reachy_mini.media.audio_gstreamer.get_audio_device",
        return_value="dummy-card",
    ) as get_dev, patch(
        "reachy_mini.media.audio_gstreamer.Gst.ElementFactory.find",
        return_value=object(),
    ):
        result = GStreamerAudio._resolve_sw_aec_enabled(_fake_aec_self())

    assert result is True
    # Hardware-detection helpers are bypassed when force-enabled.
    soundrc.assert_not_called()
    get_dev.assert_not_called()


def test_force_env_disabled_when_plugins_missing(monkeypatch) -> None:
    """Force-enable still bails out if `webrtcdsp` isn't packaged."""
    monkeypatch.delenv("REACHY_MINI_DISABLE_SW_AEC", raising=False)
    monkeypatch.setenv("REACHY_MINI_FORCE_SW_AEC", "1")

    with patch(
        "reachy_mini.media.audio_gstreamer.Gst.ElementFactory.find",
        return_value=None,
    ):
        result = GStreamerAudio._resolve_sw_aec_enabled(_fake_aec_self())

    assert result is False


def test_auto_disabled_when_asoundrc_present(monkeypatch) -> None:
    """Wireless `.asoundrc` always wins (XMOS loopback handles AEC)."""
    monkeypatch.delenv("REACHY_MINI_DISABLE_SW_AEC", raising=False)
    monkeypatch.delenv("REACHY_MINI_FORCE_SW_AEC", raising=False)

    with patch(
        "reachy_mini.media.audio_gstreamer.has_reachymini_asoundrc",
        return_value=True,
    ), patch(
        "reachy_mini.media.audio_gstreamer.get_audio_device"
    ) as get_dev:
        result = GStreamerAudio._resolve_sw_aec_enabled(_fake_aec_self())

    assert result is False
    get_dev.assert_not_called()


def test_auto_disabled_when_respeaker_card_detected(monkeypatch) -> None:
    """ReSpeaker USB dongle present (Lite happy path) skips SW AEC."""
    monkeypatch.delenv("REACHY_MINI_DISABLE_SW_AEC", raising=False)
    monkeypatch.delenv("REACHY_MINI_FORCE_SW_AEC", raising=False)

    with patch(
        "reachy_mini.media.audio_gstreamer.has_reachymini_asoundrc",
        return_value=False,
    ), patch(
        "reachy_mini.media.audio_gstreamer.get_audio_device",
        return_value="reachy-mini-audio-id",
    ):
        result = GStreamerAudio._resolve_sw_aec_enabled(_fake_aec_self())

    assert result is False


def test_auto_enabled_when_no_hardware_and_plugins_present(monkeypatch) -> None:
    """Sim / dev box: no hardware AEC, plugins available → enable SW AEC."""
    monkeypatch.delenv("REACHY_MINI_DISABLE_SW_AEC", raising=False)
    monkeypatch.delenv("REACHY_MINI_FORCE_SW_AEC", raising=False)

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


def test_auto_disabled_when_no_hardware_but_plugins_missing(monkeypatch) -> None:
    """No hardware AND no `webrtcdsp` packaged → fall back to no AEC (warn only)."""
    monkeypatch.delenv("REACHY_MINI_DISABLE_SW_AEC", raising=False)
    monkeypatch.delenv("REACHY_MINI_FORCE_SW_AEC", raising=False)

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
