"""Switching the NFC reader on and off at runtime.

The choice is stored with the other daemon settings (``nfc_enabled`` in
``daemon_config.json``, see :mod:`reachy_mini.daemon.startup_app_config`) so it
survives daemon restarts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from reachy_mini.daemon import startup_app_config

if TYPE_CHECKING:
    from .reader import NfcReader, NfcStatus


def switch_reader(reader: NfcReader, enabled: bool) -> NfcStatus:
    """Switch the reader on or off now, and remember it across restarts.

    The switch applies even when the choice cannot be saved: the returned
    status then says ``setting_saved=False``, so a client can tell the user it
    will not survive a restart. Blocking (stop() joins the reader thread):
    call it off the event loop.
    """
    saved = startup_app_config.set_nfc_enabled(enabled)
    if enabled:
        reader.start()
    else:
        reader.stop()
    return reader.get_status().model_copy(update={"setting_saved": saved})
