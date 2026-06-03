"""Tests for the audio output gain utility module."""

# ruff: noqa: D102

import os
from unittest.mock import patch

from reachy_mini.media.audio_gain import (
    ENV_VAR,
    MAX_GAIN_DB,
    MIN_GAIN_DB,
    db_to_linear,
    get_output_gain_db,
    get_output_gain_linear,
    set_output_gain_db,
)


class TestDbToLinear:
    """Test dB to linear conversion."""

    def test_zero_db_is_unity(self) -> None:
        assert db_to_linear(0) == 1.0

    def test_positive_3db(self) -> None:
        assert round(db_to_linear(3), 3) == 1.413

    def test_positive_6db(self) -> None:
        assert round(db_to_linear(6), 3) == 1.995

    def test_negative_6db(self) -> None:
        assert round(db_to_linear(-6), 3) == 0.501

    def test_negative_20db(self) -> None:
        assert round(db_to_linear(-20), 2) == 0.10

    def test_positive_12db(self) -> None:
        assert round(db_to_linear(12), 3) == 3.981


class TestEnvParsing:
    """Test environment variable parsing."""

    def _reset_module(self) -> None:
        """Reset the module-level cached gain."""
        import reachy_mini.media.audio_gain as mod

        with mod._lock:
            mod._gain_db = None

    def test_missing_env_returns_zero(self) -> None:
        self._reset_module()
        with patch.dict(os.environ, {}, clear=True):
            # Remove the env var if present
            os.environ.pop(ENV_VAR, None)
            assert get_output_gain_db() == 0.0

    def test_zero_env_returns_zero(self) -> None:
        self._reset_module()
        with patch.dict(os.environ, {ENV_VAR: "0"}):
            assert get_output_gain_db() == 0.0

    def test_positive_value(self) -> None:
        self._reset_module()
        with patch.dict(os.environ, {ENV_VAR: "3"}):
            assert get_output_gain_db() == 3.0

    def test_negative_value(self) -> None:
        self._reset_module()
        with patch.dict(os.environ, {ENV_VAR: "-6"}):
            assert get_output_gain_db() == -6.0

    def test_fractional_value(self) -> None:
        self._reset_module()
        with patch.dict(os.environ, {ENV_VAR: "1.5"}):
            assert get_output_gain_db() == 1.5

    def test_invalid_string_returns_zero(self) -> None:
        self._reset_module()
        with patch.dict(os.environ, {ENV_VAR: "loud"}):
            assert get_output_gain_db() == 0.0

    def test_empty_string_returns_zero(self) -> None:
        self._reset_module()
        with patch.dict(os.environ, {ENV_VAR: ""}):
            assert get_output_gain_db() == 0.0

    def test_linear_at_zero_is_unity(self) -> None:
        self._reset_module()
        with patch.dict(os.environ, {ENV_VAR: "0"}):
            assert get_output_gain_linear() == 1.0

    def test_linear_at_3db(self) -> None:
        self._reset_module()
        with patch.dict(os.environ, {ENV_VAR: "3"}):
            assert round(get_output_gain_linear(), 3) == 1.413


class TestSetGain:
    """Test runtime gain setting."""

    def _reset_module(self) -> None:
        import reachy_mini.media.audio_gain as mod

        with mod._lock:
            mod._gain_db = None

    def test_set_returns_value(self) -> None:
        self._reset_module()
        assert set_output_gain_db(3.0) == 3.0
        assert get_output_gain_db() == 3.0

    def test_clamps_above_max(self) -> None:
        self._reset_module()
        result = set_output_gain_db(100.0)
        assert result == MAX_GAIN_DB

    def test_clamps_below_min(self) -> None:
        self._reset_module()
        result = set_output_gain_db(-100.0)
        assert result == MIN_GAIN_DB

    def test_set_zero(self) -> None:
        self._reset_module()
        set_output_gain_db(5.0)
        set_output_gain_db(0.0)
        assert get_output_gain_db() == 0.0
