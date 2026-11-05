"""The NFC reader as the REST and RPC APIs see it: one set of rules for both.

Both APIs serve the same reader with the same refusals; only the way a refusal
is spelled differs (HTTP 503 for REST, ``nfc_disabled`` for JSON-RPC). The
rules live here so the two cannot drift apart, and each API only translates
:class:`NfcDisabledError` into its own error.

The on/off choice is stored with the other daemon settings (``nfc_enabled`` in
``daemon_config.json``, see :mod:`reachy_mini.daemon.startup_app_config`) so it
survives daemon restarts. Lives in the daemon rather than in ``reachy_mini.nfc``
so the reader package does not depend on the daemon.
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


# Saving and applying are two steps: without this, two switches at once (REST
# and RPC, or a double tap) could leave the reader running with "off" saved.
_switch_lock = threading.Lock()


def switch_reader(reader: NfcReader, enabled: bool) -> NfcStatus:
    """Switch the reader on or off now, and remember it across restarts.

    The switch applies even when the choice cannot be saved: the returned
    status then says ``setting_saved=False``, so a client can tell the user it
    will not survive a restart. Blocking (stop() joins the reader thread):
    call it off the event loop.
    """
    with _switch_lock:
        saved = startup_app_config.set_nfc_enabled(enabled)
        if enabled:
            reader.start()
        else:
            reader.stop()
        return reader.get_status().model_copy(update={"setting_saved": saved})
