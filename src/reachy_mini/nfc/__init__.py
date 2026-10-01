"""Optional NFC reader accessory (CLRC663 board, driven by ``winnie_nfc``)."""

from .ports import exclude_nfc_boards, find_nfc_port
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
    driver_available,
    no_tag,
)
from .settings import get_nfc_enabled, set_nfc_enabled

__all__ = [
    "NfcDump",
    "NfcEnableRequest",
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
    "get_nfc_enabled",
    "no_tag",
    "set_nfc_enabled",
]
