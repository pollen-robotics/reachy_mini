"""Persisted daemon config for the startup app.

The startup app (launched on the robot's first wake-up) is stored in a small
JSON file in the user's config dir, so the choice survives reboots and app
updates, stays per-user (not shared across OS accounts on one machine), and can
be set over the REST API instead of only via a CLI flag.
"""

import json
import logging
import math
from pathlib import Path

import platformdirs

logger = logging.getLogger(__name__)

_KEY = "startup_app"
_EQ_KEY = "speaker_eq_gains"
# equalizer-10bands accepts per-band gains in [-24, +12] dB.
_EQ_GAIN_MIN, _EQ_GAIN_MAX = -24.0, 12.0


def _is_valid_gain(value: object) -> bool:
    """Return True for a finite real number within the equalizer dB range."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and _EQ_GAIN_MIN <= value <= _EQ_GAIN_MAX
    )


def _config_path() -> Path:
    """Path to the daemon config file in the user's config dir."""
    return Path(platformdirs.user_config_dir("reachy_mini")) / "daemon_config.json"


def _read() -> dict:  # type: ignore[type-arg]
    """Load the config dict, or {} if missing/unreadable (best-effort)."""
    path = _config_path()
    try:
        with path.open() as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Ignoring unreadable daemon config {path}: {e}")
        return {}


def get_startup_app() -> str | None:
    """Return the persisted startup app name, or None if unset."""
    value = _read().get(_KEY)
    return value if isinstance(value, str) else None


def get_speaker_eq_gains() -> list[float] | None:
    """Return the 10 speaker-EQ band gains (dB), or None if unset/invalid.

    Invalid values (wrong length, non-numeric, NaN/inf, or outside the
    equalizer-10bands [-24, +12] dB range) are treated as unset so the caller
    falls back to its built-in default.
    """
    config = _read()
    if _EQ_KEY not in config:
        return None
    value = config[_EQ_KEY]
    if (
        isinstance(value, list)
        and len(value) == 10
        and all(_is_valid_gain(x) for x in value)
    ):
        return [float(x) for x in value]
    # Present but malformed: warn so the user knows their values were ignored.
    logger.warning(
        "Ignoring invalid '%s' in daemon config (need 10 finite dB gains in "
        "[%g, %g]); using the built-in defaults.",
        _EQ_KEY,
        _EQ_GAIN_MIN,
        _EQ_GAIN_MAX,
    )
    return None


def set_startup_app(name: str | None) -> None:
    """Persist the startup app name; a falsy name clears it."""
    config = _read()
    if name:
        config[_KEY] = name
    else:
        config.pop(_KEY, None)

    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(config, f, indent=2)
