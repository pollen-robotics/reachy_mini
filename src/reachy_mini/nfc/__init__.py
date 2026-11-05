"""Optional NFC reader accessory (CLRC663 board, driven by ``winnie_nfc``)."""

from .ports import NfcBoard, find_nfc_board, is_nfc_board
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
    driver_available,
    no_tag,
)

__all__ = [
    "DISABLED_ERROR",
    "NfcBoard",
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
    "driver_available",
    "find_nfc_board",
    "is_nfc_board",
    "no_tag",
]
