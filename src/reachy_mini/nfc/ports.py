"""Serial port discovery for the NFC reader board.

The board is recognised by its USB product string, ``Winnie NFC <version>``.
Reading it is left to the driver, ``winnie_nfc.usb``: pyserial reports it on
Linux and macOS, and the driver reads it from the OS on Windows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import ModuleType

logger = logging.getLogger(__name__)

# USB product string of the NFC board, followed by its version: "Winnie NFC 1.0".
NFC_PRODUCT_PREFIX = "Winnie NFC"


@dataclass(frozen=True)
class NfcBoard:
    """An NFC reader board found on a serial port."""

    port: str
    version: str | None  # from the USB product string; None if it has none


def parse_product(product: str | None) -> tuple[bool, str | None]:
    """Tell whether a USB product string is the NFC board's, and its version.

    Returns ``(is_nfc_board, version)``. The version is whatever follows the
    prefix, surrounding blanks removed: ``"Winnie NFC 1.2"`` gives ``"1.2"``;
    a bare ``"Winnie NFC"`` is the board with no version.
    """
    if not product:
        return False, None
    product = product.strip()
    if not product.casefold().startswith(NFC_PRODUCT_PREFIX.casefold()):
        return False, None
    version = product[len(NFC_PRODUCT_PREFIX) :].strip()
    return True, version or None


def _usb() -> ModuleType | None:
    """Return the driver's USB module, or None when it cannot be imported.

    Port lookup must never take the daemon down: the motor port search goes
    through here too (see ``daemon.utils.find_serial_port``).
    """
    try:
        from winnie_nfc import usb
    except Exception as e:  # noqa: BLE001 - a broken install reads as no board
        logger.debug("winnie_nfc unavailable, no NFC board lookup: %s", e)
        return None
    module: ModuleType = usb  # untyped package: name what it is
    return module


def nfc_board_ports() -> list[str]:
    """Return the ports of every NFC reader board plugged in, sorted."""
    usb = _usb()
    if usb is None:
        return []
    found: list[str] = usb.winnie_ports()
    return found


def find_nfc_board() -> NfcBoard | None:
    """Return the NFC reader board, or ``None`` if there is none.

    With several boards plugged in, the first port in sort order.
    """
    found = nfc_board_ports()
    if len(found) > 1:
        logger.warning(
            "Several NFC reader boards found (%s), using %s.", found, found[0]
        )
    return describe_port(found[0]) if found else None


def describe_port(port: str) -> NfcBoard:
    """Describe a port, with the board version if its product string has one."""
    usb = _usb()
    info = usb.port_info(port) if usb is not None else None
    product: str | None = getattr(info, "product", None)
    return NfcBoard(port=port, version=parse_product(product)[1])
