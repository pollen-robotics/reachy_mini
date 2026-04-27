"""Persistent daemon configuration.

Holds a tiny JSON document under ``$XDG_CONFIG_HOME/reachy_mini/daemon.json``
(default ``~/.config/reachy_mini/daemon.json``) for daemon-level state that
must survive restarts and re-installs of the upstream tray app.

Currently the only persisted field is ``robot_name`` - the human-readable
label the user picked for this Reachy. We deliberately keep this file
separate from any tray ``data_dir`` so that:

- the same physical Reachy keeps its name across a tray ``reset_bootstrap``,
- the same daemon code path applies on the Wireless robot (where there is
  no tray at all) and on the Lite/desktop tray on macOS or Linux,
- destroying the HF token cache does not nuke the user's robot label.

Reads and writes are tolerant: a missing or corrupt file is treated as
"no persisted config", and a write failure is logged but never raises.
The naming flow always has a mandatory in-memory fallback (``reachy_mini``).

Concurrency: writes are ``fsync`` + atomic rename, so a crash mid-write
cannot leave the config in a half-written state. There is no in-process
locking here; the caller (single FastAPI worker) is responsible for not
issuing parallel ``save_config`` calls. The HTTP route that mutates the
name guards itself with an ``asyncio.Lock``.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = "daemon.json"

# Hard-coded fallback used when the user has not yet customised their
# robot's name. Kept here so every layer of the daemon (config loader,
# resolver in main.py, central-relay gate in daemon.py, rename HTTP route)
# agrees on the same string and a single rename touches one place.
DEFAULT_ROBOT_NAME = "reachy_mini"


def _config_dir() -> Path:
    """Resolve the directory that holds ``daemon.json``.

    Honours ``$XDG_CONFIG_HOME`` when set (used by both Linux conventions
    and the macOS tray when it wants a custom location), otherwise falls
    back to ``~/.config`` per the XDG Base Directory spec. The ``reachy_mini``
    subfolder keeps daemon config separate from any other Reachy tooling.
    """
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "reachy_mini"
    return Path.home() / ".config" / "reachy_mini"


def get_config_path() -> Path:
    """Return the absolute path to ``daemon.json`` (file may not exist)."""
    return _config_dir() / _CONFIG_FILENAME


def load_config() -> dict[str, Any]:
    """Load the persisted daemon config.

    Returns an empty dict when the file is missing or unreadable. Never
    raises: corrupted config must not prevent the daemon from booting.
    """
    path = get_config_path()
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                return data
            logger.warning("[config] %s does not contain a JSON object, ignoring", path)
            return {}
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[config] failed to read %s: %s", path, exc)
        return {}


def save_config(config: dict[str, Any]) -> bool:
    """Persist ``config`` atomically.

    Writes to a temp file in the same directory, fsyncs it, then renames
    over the target. Returns ``True`` on success, ``False`` on failure.
    Never raises.
    """
    path = get_config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("[config] failed to mkdir %s: %s", path.parent, exc)
        return False

    try:
        # ``delete=False`` so we keep control of the rename ourselves.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".daemon.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            json.dump(config, tmp, indent=2, sort_keys=True)
            tmp.write("\n")
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
    except OSError as exc:
        logger.warning("[config] failed to write tmp file in %s: %s", path.parent, exc)
        return False

    try:
        os.replace(tmp_path, path)
    except OSError as exc:
        logger.warning(
            "[config] failed to atomically rename %s -> %s: %s", tmp_path, path, exc
        )
        try:
            tmp_path.unlink()
        except OSError:
            pass
        return False

    return True


def get_persisted_robot_name() -> Optional[str]:
    """Return the previously saved robot name, or ``None`` if unset."""
    name = load_config().get("robot_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def set_persisted_robot_name(name: str) -> bool:
    """Persist ``name`` as the robot name. Returns whether the write succeeded.

    Reads-modifies-writes the config so we don't clobber sibling fields
    that future versions may add.
    """
    config = load_config()
    config["robot_name"] = name
    return save_config(config)
