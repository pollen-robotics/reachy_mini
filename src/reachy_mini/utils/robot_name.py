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

logger = logging.getLogger(__name__)

_STATE_PATH = Path.home() / ".config" / "reachy_mini" / "robot_name.json"

# Keep names short enough for mDNS / UI labels and free of surrounding
# whitespace. Empty names are rejected (treated as "no override").
_MAX_NAME_LENGTH = 64


def get_robot_name() -> str | None:
    """Return the persisted robot name, or None if none is stored.

    Defaults to None (no override) on a missing file or any read / parse
    error, so the caller falls back to the CLI default.
    """
    try:
        data = json.loads(_STATE_PATH.read_text())
        name = data.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        return None
    except FileNotFoundError:
        return None
    except Exception as e:  # noqa: BLE001 - never crash the command loop
        logger.warning("Could not read robot name: %s", e)
        return None


def set_robot_name(name: str) -> str | None:
    """Persist a new robot name.

    Returns the stored (trimmed, length-capped) name on success, or None on
    empty/invalid input or a write error.
    """
    sanitized = name.strip()[:_MAX_NAME_LENGTH] if isinstance(name, str) else ""
    if not sanitized:
        return None
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps({"name": sanitized}))
        return sanitized
    except Exception as e:  # noqa: BLE001 - never crash the command loop
        logger.warning("Could not write robot name: %s", e)
        return None
