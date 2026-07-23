"""Persistence for the first wake-up setup wizard flag.

The mobile / desktop apps run a one-time, post-connection hardware
diagnostic wizard ("first wake-up"). Whether it has been completed is a
persistent, robot-wide boolean stored on the robot itself, so the wizard
only ever shows once regardless of which client connects.

It lives as a tiny JSON file under the user's config dir. Every failure
path is swallowed and treated as "not completed" / "write failed" so a
read/write error can never crash the daemon command loop.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_STATE_PATH = Path.home() / ".config" / "reachy_mini" / "first_wake_up.json"


def get_first_wake_up_completed() -> bool:
    """Return True if the first wake-up wizard has been completed.

    Defaults to False (wizard pending) on a missing file or any read /
    parse error.
    """
    try:
        data = json.loads(_STATE_PATH.read_text())
        return bool(data.get("is_completed", False))
    except FileNotFoundError:
        return False
    except Exception as e:  # noqa: BLE001 - never crash the command loop
        logger.warning("Could not read first-wake-up state: %s", e)
        return False


def set_first_wake_up_completed(is_completed: bool) -> bool:
    """Persist the first wake-up completion flag. Returns True on success."""
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps({"is_completed": bool(is_completed)}))
        return True
    except Exception as e:  # noqa: BLE001 - never crash the command loop
        logger.warning("Could not write first-wake-up state: %s", e)
        return False
