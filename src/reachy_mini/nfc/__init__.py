"""Optional NFC reader accessory (CLRC663 board, driven by ``winnie_nfc``).

The daemon owns the reader: applications reach it through the daemon's
``/api/nfc`` routes or its ``nfc.*`` JSON-RPC methods, never by opening the
serial port themselves. What is exported here is what the daemon serves.
"""

from .reader import (
    DISABLED_ERROR,
    NfcDump,
    NfcEnableRequest,
    NfcEraseRequest,
    NfcReader,
    NfcRecord,
    NfcStatus,
    NfcStatusError,
    NfcTag,
    NfcWriteRequest,
    NfcWriteResult,
    disabled_status,
    no_tag,
)

__all__ = [
    "DISABLED_ERROR",
    "NfcDump",
    "NfcEnableRequest",
    "NfcEraseRequest",
    "NfcReader",
    "NfcRecord",
    "NfcStatus",
    "NfcStatusError",
    "NfcTag",
    "NfcWriteRequest",
    "NfcWriteResult",
    "disabled_status",
    "no_tag",
]
