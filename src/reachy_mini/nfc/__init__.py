"""Optional NFC reader accessory (Arduino + PN532 over USB serial)."""

from .reader import (
    NfcReader,
    NfcStatus,
    NfcTag,
    NfcWriteRequest,
    NfcWriteResult,
    find_nfc_ports,
)

__all__ = [
    "NfcReader",
    "NfcStatus",
    "NfcTag",
    "NfcWriteRequest",
    "NfcWriteResult",
    "find_nfc_ports",
]
