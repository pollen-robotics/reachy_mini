"""Optional NFC reader accessory (CLRC663 board, driven by ``winnie_nfc``)."""

from .ports import NfcBoard, find_nfc_board, is_nfc_board
from .reader import (
    NfcDump,
    NfcEnableRequest,
    NfcEraseRequest,
    NfcReader,
    NfcRecord,
    NfcStatus,
    NfcTag,
    NfcWriteRequest,
    NfcWriteResult,
    disabled_status,
    driver_available,
    no_tag,
)
from .settings import switch_reader

__all__ = [
    "NfcBoard",
    "NfcDump",
    "NfcEnableRequest",
    "NfcEraseRequest",
    "NfcReader",
    "NfcRecord",
    "NfcStatus",
    "NfcTag",
    "NfcWriteRequest",
    "NfcWriteResult",
    "disabled_status",
    "driver_available",
    "find_nfc_board",
    "is_nfc_board",
    "no_tag",
    "switch_reader",
]
