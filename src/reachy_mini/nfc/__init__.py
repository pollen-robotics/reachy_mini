"""Optional NFC reader accessory (CLRC663 board, driven by ``winnie_nfc``)."""

from .ports import exclude_nfc_boards, find_nfc_port
from .reader import (
    NfcDump,
    NfcEraseRequest,
    NfcReader,
    NfcRecord,
    NfcStatus,
    NfcTag,
    NfcWriteRequest,
    NfcWriteResult,
    driver_available,
    no_tag,
)

__all__ = [
    "NfcDump",
    "NfcEraseRequest",
    "NfcReader",
    "NfcRecord",
    "NfcStatus",
    "NfcTag",
    "NfcWriteRequest",
    "NfcWriteResult",
    "driver_available",
    "exclude_nfc_boards",
    "find_nfc_port",
    "no_tag",
]
