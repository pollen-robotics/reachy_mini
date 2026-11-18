"""Persistence for the user-set robot name.

The robot's display name (advertised to the central signaling relay, over
mDNS, and shown in the apps' robot list) defaults to the daemon's
``--robot-name`` CLI argument. When a client renames the robot (e.g. from
the mobile app over the WebRTC data channel) the new name is stored here so
it survives reboots and wins over the CLI default at the next daemon start.

It lives as a tiny JSON file under the user's config dir, next to the
first-wake-up flag. Every failure path is swallowed so a read/write error
can never crash the daemon command loop.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import platformdirs

logger = logging.getLogger(__name__)

_STATE_PATH = Path(platformdirs.user_config_dir("reachy_mini")) / "robot_name.json"
# Where it lived before; differs from _STATE_PATH on macOS/Windows or with
# XDG_CONFIG_HOME set. Read-only fallback, the next rename writes _STATE_PATH.
_LEGACY_PATH = Path.home() / ".config" / "reachy_mini" / "robot_name.json"

# Keep names short enough for mDNS / UI labels and free of surrounding
# whitespace. Empty names are rejected (treated as "no override"). This is the
# single source of truth for the cap; the SetRobotNameCmd schema imports it.
MAX_ROBOT_NAME_LENGTH = 64


def get_robot_name() -> str | None:
    """Return the persisted robot name, or None if none is stored.

    Defaults to None (no override) on a missing file or any read / parse
    error, so the caller falls back to the CLI default.
    """
    for path in (_STATE_PATH, _LEGACY_PATH):
        try:
            data = json.loads(path.read_text())
            name = data.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
            return None
        except FileNotFoundError:
            continue
        except Exception as e:  # noqa: BLE001 - never crash the command loop
            logger.warning("Could not read robot name: %s", e)
            return None
    return None


def set_robot_name(name: str) -> str | None:
    """Persist a new robot name.

    Returns the stored (trimmed, length-capped) name on success, or None on
    empty/invalid input or a write error.
    """
    sanitized = name.strip()[:MAX_ROBOT_NAME_LENGTH] if isinstance(name, str) else ""
    if not sanitized:
        return None
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps({"name": sanitized}))
        return sanitized
    except Exception as e:  # noqa: BLE001 - never crash the command loop
        logger.warning("Could not write robot name: %s", e)
        return None
