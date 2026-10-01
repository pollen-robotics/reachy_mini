"""Persistence for the NFC reader on/off switch.

The reader is on by default. When a client switches it off (e.g. from the
desktop app's Settings) the choice is stored here so it survives daemon
restarts. ``--no-nfc`` still wins: it removes the reader altogether.

It lives as a tiny JSON file under the user's config dir, next to the robot
name. Every failure path is swallowed: a read or write error must never stop
the daemon, it only loses the setting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_STATE_PATH = Path.home() / ".config" / "reachy_mini" / "nfc.json"


def get_nfc_enabled() -> bool:
    """Return whether the reader is switched on, True when never set."""
    try:
        data = json.loads(_STATE_PATH.read_text())
        enabled = data.get("enabled")
        return enabled if isinstance(enabled, bool) else True
    except FileNotFoundError:
        return True
    except Exception as e:  # noqa: BLE001 - never stop the daemon over this
        logger.warning("Could not read the NFC setting: %s", e)
        return True


def set_nfc_enabled(enabled: bool) -> bool:
    """Persist the switch. Returns False if it could not be written."""
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps({"enabled": enabled}))
        return True
    except Exception as e:  # noqa: BLE001 - never stop the daemon over this
        logger.warning("Could not write the NFC setting: %s", e)
        return False
