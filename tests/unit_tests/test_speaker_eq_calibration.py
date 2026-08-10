"""Unit tests for the offline speaker-EQ calibration band math."""

import numpy as np
import pytest

from reachy_mini.tools.speaker_eq_calibration.calibrate import (
    BAND_CENTERS,
    GAIN_MAX,
    GAIN_MIN,
    band_gains_from_responses,
)

# Dense linear freq axis up to 20 kHz so every band window is populated.
FREQS = np.linspace(0.0, 20_000.0, 20_001)


def _band_mask(center: float) -> np.ndarray:
    return (FREQS >= center / 2**0.5) & (FREQS < center * 2**0.5)


def test_shell_bump_becomes_matching_cut() -> None:
    """A shell resonance in one band maps to an equal cut in that band."""
    # Shell-on response is 6 dB hotter than shell-off around the 237 Hz band
    # (the shell adds a boxy bump) -> the EQ should cut 6 dB there.
    resp_off = np.zeros_like(FREQS)
    resp_on = np.zeros_like(FREQS)
    resp_on[_band_mask(237.0)] = +6.0

    gains = band_gains_from_responses(FREQS, resp_on, resp_off, zero_above=8000.0)

    idx = BAND_CENTERS.index(237.0)
    assert gains[idx] == pytest.approx(-6.0, abs=0.5)


def test_out_of_range_is_clamped() -> None:
    """Gains outside the element's range are clamped to [GAIN_MIN, GAIN_MAX]."""
    resp_off = np.zeros_like(FREQS)
    resp_on = np.zeros_like(FREQS)
    resp_on[_band_mask(474.0)] = +100.0  # huge shell bump -> huge cut
    resp_on[_band_mask(947.0)] = -100.0  # huge shell dip -> huge boost

    gains = band_gains_from_responses(FREQS, resp_on, resp_off, zero_above=8000.0)

    assert gains[BAND_CENTERS.index(474.0)] == GAIN_MIN
    assert gains[BAND_CENTERS.index(947.0)] == GAIN_MAX
    assert all(GAIN_MIN <= g <= GAIN_MAX for g in gains)


def test_zero_above_and_max_boost() -> None:
    """Bands above zero_above are forced to 0 dB and boosts are capped."""
    resp_off = np.zeros_like(FREQS)
    resp_on = np.zeros_like(FREQS)
    resp_on[_band_mask(474.0)] = -20.0  # shell dip -> wants a big +20 boost

    gains = band_gains_from_responses(
        FREQS, resp_on, resp_off, zero_above=3000.0, max_boost=6.0
    )

    assert gains[BAND_CENTERS.index(474.0)] == pytest.approx(6.0)  # capped
    for center, gain in zip(BAND_CENTERS, gains):
        if center > 3000.0:
            assert gain == 0.0
