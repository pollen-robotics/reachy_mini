"""Persisted daemon config for the startup app.

The startup app (launched on the robot's first wake-up) is stored in a small
JSON file in the user's config dir, so the choice survives reboots and app
updates, stays per-user (not shared across OS accounts on one machine), and can
be set over the REST API instead of only via a CLI flag.
"""

import json
import logging
from pathlib import Path

import platformdirs

logger = logging.getLogger(__name__)

_KEY = "startup_app"
_EQ_KEY = "speaker_eq_gains"
_TURN_KEY = "turn_enabled"
_FIRST_WAKE_UP_KEY = "first_wake_up_completed"
# equalizer-10bands accepts per-band gains in [-24, +12] dB.
_EQ_GAIN_MIN, _EQ_GAIN_MAX = -24.0, 12.0


def _is_valid_gain(value: object) -> bool:
    """Return True for a real number within the equalizer dB range.

    The range comparison also rejects NaN and infinities (they compare False)
    and oversized ints (exact int/float compare, so no OverflowError) without
    converting the value.
    """
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and _EQ_GAIN_MIN <= value <= _EQ_GAIN_MAX
    )


def _config_path() -> Path:
    """Path to the daemon config file in the user's config dir."""
    return Path(platformdirs.user_config_dir("reachy_mini")) / "daemon_config.json"


def _read() -> dict:  # type: ignore[type-arg]
    """Load the config dict, or {} if missing/unreadable (best-effort)."""
    path = _config_path()
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
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


def _get_bool(key: str) -> bool | None:
    """Return the boolean at `key`, or None when unset or malformed.

    A present-but-malformed value is warned about (so the user knows their
    hand-edited config was ignored) and treated as unset.
    """
    config = _read()
    if key not in config:
        return None
    value = config[key]
    if isinstance(value, bool):
        return value
    logger.warning(
        "Ignoring invalid '%s' in daemon config (need true or false); "
        "using the built-in default.",
        key,
    )
    return None


def get_turn_enabled() -> bool | None:
    """Return whether the media server should offer TURN relay candidates.

    Returns None if unset or malformed, so the caller keeps its default. A
    relay lets a remote consumer behind a restrictive NAT reach the robot;
    turning it off saves a background thread refreshing credentials on a
    robot that is only ever reached from its own network.
    """
    return _get_bool(_TURN_KEY)


def _update(key: str, value: object | None) -> None:
    """Read-modify-write a single config key; `None` clears it.

    Other keys are preserved. May raise OSError on a write error; callers
    that must never raise (e.g. the daemon command loop) wrap it themselves.
    """
    config = _read()
    if value is None:
        config.pop(key, None)
    else:
        config[key] = value

    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling then rename: a crash mid-write would otherwise
    # truncate the file, and the next _update() rebuilds it from the empty
    # dict _read() returns for unparseable JSON, dropping every other key.
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    tmp.replace(path)


def set_startup_app(name: str | None) -> None:
    """Persist the startup app name; a falsy name clears it."""
    _update(_KEY, name if name else None)


def get_first_wake_up_completed() -> bool:
    """Return True if the first wake-up setup wizard has been completed.

    Robot-wide, persistent flag (not per-session): the mobile / desktop apps
    run a one-time, post-connection hardware diagnostic wizard, and gate it on
    this so it only ever shows once, whichever client connects. Defaults to
    False (wizard pending) when unset or malformed, so a config problem can
    never trap the daemon command loop.
    """
    return _get_bool(_FIRST_WAKE_UP_KEY) is True


def set_first_wake_up_completed(is_completed: bool) -> bool:
    """Persist the first wake-up completion flag. Returns True on success.

    Fail-safe: a write error is logged and reported as False instead of
    raising, so a storage problem can't break the daemon command loop.
    """
    try:
        _update(_FIRST_WAKE_UP_KEY, bool(is_completed))
        return True
    except OSError as e:
        logger.warning(f"Could not persist first-wake-up flag: {e}")
        return False
