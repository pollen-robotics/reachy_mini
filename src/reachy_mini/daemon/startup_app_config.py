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
