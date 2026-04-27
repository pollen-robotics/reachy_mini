"""Tests for Reachy Mini audio control helpers."""

import pytest

from reachy_mini.media.audio_control_utils import PARAMETERS, init_respeaker_usb
from reachy_mini.media.media_manager import MediaBackend, MediaManager

AUDIO_CONFIG_PARAMETER_NAMES = ("PP_MIN_NS", "PP_NLATTENONOFF", "PP_MGSCALE")


@pytest.mark.audio
def test_respeaker_read_values_reads_board_parameters() -> None:
    """Numeric readback should be normalized from the real audio board."""
    respeaker = init_respeaker_usb()
    assert respeaker is not None, "Reachy Mini Audio board is required."
    try:
        for name in AUDIO_CONFIG_PARAMETER_NAMES:
            values = respeaker.read_values(name)
            assert values is not None
            assert len(values) == PARAMETERS[name][2]
            assert all(isinstance(value, (float, int)) for value in values)
    finally:
        respeaker.close()


@pytest.mark.audio
def test_respeaker_apply_audio_config_writes_current_board_values() -> None:
    """Custom config writes should be verified against real board readback."""
    respeaker = init_respeaker_usb()
    assert respeaker is not None, "Reachy Mini Audio board is required."
    try:
        config = []
        for name in AUDIO_CONFIG_PARAMETER_NAMES:
            values = respeaker.read_values(name)
            assert values is not None
            config.append((name, values))

        assert respeaker.apply_audio_config(tuple(config))
        for name, expected_values in config:
            assert respeaker.read_values(name) == pytest.approx(expected_values)
    finally:
        respeaker.close()


@pytest.mark.audio
def test_media_audio_apply_audio_config_uses_real_board() -> None:
    """Media audio should apply caller-provided config through the real board."""
    respeaker = init_respeaker_usb()
    assert respeaker is not None, "Reachy Mini Audio board is required."
    try:
        values = respeaker.read_values("PP_MIN_NS")
        assert values is not None
    finally:
        respeaker.close()

    media = MediaManager(backend=MediaBackend.LOCAL)
    try:
        assert media.audio.apply_audio_config((("PP_MIN_NS", values),))
    finally:
        media.close()
