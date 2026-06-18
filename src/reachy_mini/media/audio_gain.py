"""Daemon-level output gain configuration.

Reads the ``REACHY_AUDIO_GAIN_DB`` environment variable and provides
a linear multiplier suitable for a GStreamer ``volume`` element.

Usage::

    from reachy_mini.media.audio_gain import get_output_gain_linear, get_output_gain_db

    linear = get_output_gain_linear()  # e.g. 1.412 for +3 dB
    db = get_output_gain_db()          # e.g. 3.0
"""

import logging
import os
from threading import Lock

_logger = logging.getLogger(__name__)

ENV_VAR = "REACHY_AUDIO_GAIN_DB"

# Safe range for runtime API changes (dB).
MIN_GAIN_DB = -20.0
MAX_GAIN_DB = 24.0

_lock = Lock()
_gain_db: float | None = None  # lazily initialized from env


def db_to_linear(db: float) -> float:
    """Convert decibels to a linear multiplier.

    >>> db_to_linear(0)
    1.0
    >>> round(db_to_linear(3), 3)
    1.413
    >>> round(db_to_linear(-6), 3)
    0.501
    """
    return 10 ** (db / 20.0)


def _read_env() -> float:
    """Read and parse the environment variable. Returns 0.0 on missing/invalid."""
    raw = os.environ.get(ENV_VAR)
    if raw is None:
        return 0.0
    try:
        value = float(raw)
    except (ValueError, TypeError):
        _logger.warning(
            f"{ENV_VAR}={raw!r} is not a valid number; defaulting to 0 dB."
        )
        return 0.0
    if value > MAX_GAIN_DB:
        _logger.warning(
            f"{ENV_VAR}={value} dB exceeds recommended maximum ({MAX_GAIN_DB} dB). "
            "Clipping may occur."
        )
    return value


def get_output_gain_db() -> float:
    """Return the current output gain in dB (thread-safe)."""
    global _gain_db
    with _lock:
        if _gain_db is None:
            _gain_db = _read_env()
        return _gain_db


def set_output_gain_db(db: float) -> float:
    """Set the output gain in dB, clamped to the safe range.

    Returns the clamped value actually applied.
    """
    global _gain_db
    clamped = max(MIN_GAIN_DB, min(MAX_GAIN_DB, db))
    with _lock:
        _gain_db = clamped
    return clamped


def get_output_gain_linear() -> float:
    """Return the current output gain as a linear multiplier."""
    return db_to_linear(get_output_gain_db())
