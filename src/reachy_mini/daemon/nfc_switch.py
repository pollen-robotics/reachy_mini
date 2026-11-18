"""Access rules to the NFC reader, shared by the REST and JSON-RPC APIs.

Each API translates :class:`NfcDisabledError` into its own error (HTTP 503,
``nfc_disabled``). The on/off switch is persisted as ``nfc_enabled`` in
``daemon_config.json``.
"""

from __future__ import annotations

import threading

from reachy_mini.daemon import startup_app_config
from reachy_mini.nfc import (
    DISABLED_ERROR,
    NfcReader,
    NfcStatus,
    NfcTag,
    disabled_status,
    no_tag,
)


class NfcDisabledError(Exception):
    """The reader is switched off, or the daemon runs without one (--no-nfc)."""

    def __init__(self) -> None:
        """Carry the same message as the status of a disabled reader."""
        super().__init__(DISABLED_ERROR)


def status(reader: NfcReader | None) -> NfcStatus:
    """Return the reader status; always answers, even without a reader."""
    return reader.get_status() if reader is not None else disabled_status()


def tag(reader: NfcReader | None) -> NfcTag:
    """Return the tag on the reader; always answers, even without a reader."""
    return reader.get_tag() if reader is not None else no_tag()


def require_reader(reader: NfcReader | None) -> NfcReader:
    """Return the reader, or refuse when the daemon runs with ``--no-nfc``."""
    if reader is None:
        raise NfcDisabledError()
    return reader


def require_enabled(reader: NfcReader | None) -> NfcReader:
    """Return the reader, or refuse a tag operation while it is off."""
    if reader is None or not reader.is_enabled():
        raise NfcDisabledError()
    return reader


# Saving and applying the switch happen together.
_switch_lock = threading.Lock()


def switch_reader(reader: NfcReader, enabled: bool) -> NfcStatus:
    """Switch the reader on or off now, and remember it across restarts.

    Applies even when the choice cannot be saved (``setting_saved=False``).
    Blocking: call it off the event loop.
    """
    with _switch_lock:
        saved = startup_app_config.set_nfc_enabled(enabled)
        if enabled:
            reader.start()
        else:
            reader.stop()
        return reader.get_status().model_copy(update={"setting_saved": saved})
