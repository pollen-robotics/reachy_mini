"""Serial port discovery for the NFC reader board.

The NFC reader (the Winnie board, a CLRC663 driven by ``winnie_nfc``) is
recognised by its USB product string alone: ``Winnie NFC <version>``. The
version is read from it at the same time, at no cost: the OS has it already,
nothing is sent to the board.

pyserial reports the product string on Linux and macOS. On Windows it does not,
so it is read from the device's bus-reported description through SetupAPI.

This module is imported by ``daemon.utils.find_serial_port`` as well, to keep
the NFC board out of the motor controller's candidate list.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

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
    prefix, brackets and surrounding blanks removed: ``"Winnie NFC 1.2"`` and
    ``"Winnie NFC [1.2]"`` both give ``"1.2"``; a bare ``"Winnie NFC"`` is the
    board with no version.
    """
    if not product:
        return False, None
    product = product.strip()
    if not product.casefold().startswith(NFC_PRODUCT_PREFIX.casefold()):
        return False, None
    version = product[len(NFC_PRODUCT_PREFIX) :].strip().strip("[]()").strip()
    return True, version or None


def port_product(comport: Any) -> str | None:
    """Return the USB product string of a ``comports()`` entry, if the OS reports it."""
    product: str | None = getattr(comport, "product", None)
    if product is None and sys.platform == "win32" and getattr(comport, "vid", None):
        product = _windows_product(comport.device)
    return product


def is_nfc_board(comport: Any) -> bool:
    """Whether a ``comports()`` entry is an NFC reader board."""
    return parse_product(port_product(comport))[0]


def _comports() -> list[Any]:
    import serial.tools.list_ports

    return list(serial.tools.list_ports.comports())


def find_nfc_board(
    comports: Sequence[Any] | None = None,
    exclude: Iterable[str] = (),
) -> NfcBoard | None:
    """Return the NFC reader board, or ``None`` if there is none.

    Args:
        comports: optional list of ports (for testing); defaults to the live
            ``serial.tools.list_ports.comports()``.
        exclude: ports to leave out (typically the motor controller's port,
            when it is known explicitly).

    """
    excluded = set(exclude)
    for p in _comports() if comports is None else comports:
        if p.device in excluded:
            continue
        found, version = parse_product(port_product(p))
        if found:
            return NfcBoard(port=p.device, version=version)
    return None


def describe_port(port: str, comports: Sequence[Any] | None = None) -> NfcBoard:
    """Describe an explicitly configured port, reading its version if it has one.

    Used when the port is given rather than looked up: the board is trusted to
    be there, and only its version is wanted.
    """
    for p in _comports() if comports is None else comports:
        if p.device == port:
            return NfcBoard(port=port, version=parse_product(port_product(p))[1])
    return NfcBoard(port=port, version=None)


def _windows_product(device: str) -> str | None:
    """Read the USB product string of the device behind a COM port, on Windows.

    pyserial leaves ``product`` empty there: Windows names ports after their
    driver ("USB Serial Device (COM5)"), and keeps what the device itself
    reported as its bus-reported description. That is the product string on
    the USB device node; with the inbox CDC driver the COM port sits on an
    interface node below it, so the parents are looked at too.
    """
    try:
        return _windows_product_unsafe(device)
    except Exception as e:  # noqa: BLE001 - a missing name is not worth a crash
        logger.debug("Cannot read the USB product string of %s: %s", device, e)
        return None


def _windows_product_unsafe(device: str) -> str | None:
    import ctypes
    import uuid
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", wintypes.BYTE * 8),
        ]

    class SP_DEVINFO_DATA(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("ClassGuid", GUID),
            ("DevInst", wintypes.DWORD),
            ("Reserved", ctypes.c_void_p),
        ]

    class DEVPROPKEY(ctypes.Structure):
        _fields_ = [("fmtid", GUID), ("pid", wintypes.ULONG)]

    def guid(text: str) -> GUID:
        return GUID.from_buffer_copy(uuid.UUID(text).bytes_le)

    ports_class = guid("4d36e978-e325-11ce-bfc1-08002be10318")  # GUID_DEVCLASS_PORTS
    bus_reported_desc = DEVPROPKEY(  # DEVPKEY_Device_BusReportedDeviceDesc
        guid("540b947e-8b40-45bc-a8a2-6a0b894cbda2"), 4
    )
    DIGCF_PRESENT = 0x2
    DICS_FLAG_GLOBAL = 0x1
    DIREG_DEV = 0x1
    KEY_READ = 0x20019
    INVALID_HANDLE = ctypes.c_void_p(-1).value

    setupapi = ctypes.WinDLL("setupapi")  # type: ignore[attr-defined]
    cfgmgr = ctypes.WinDLL("cfgmgr32")  # type: ignore[attr-defined]
    advapi = ctypes.WinDLL("advapi32")  # type: ignore[attr-defined]
    setupapi.SetupDiGetClassDevsW.restype = ctypes.c_void_p
    setupapi.SetupDiGetClassDevsW.argtypes = [
        ctypes.POINTER(GUID),
        wintypes.LPCWSTR,
        wintypes.HWND,
        wintypes.DWORD,
    ]
    setupapi.SetupDiEnumDeviceInfo.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(SP_DEVINFO_DATA),
    ]
    setupapi.SetupDiOpenDevRegKey.restype = ctypes.c_void_p
    setupapi.SetupDiOpenDevRegKey.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(SP_DEVINFO_DATA),
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    setupapi.SetupDiDestroyDeviceInfoList.argtypes = [ctypes.c_void_p]
    advapi.RegQueryValueExW.argtypes = [
        ctypes.c_void_p,
        wintypes.LPCWSTR,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi.RegCloseKey.argtypes = [ctypes.c_void_p]
    cfgmgr.CM_Get_DevNode_PropertyW.argtypes = [
        wintypes.DWORD,
        ctypes.POINTER(DEVPROPKEY),
        ctypes.POINTER(wintypes.ULONG),
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.ULONG),
        wintypes.ULONG,
    ]
    cfgmgr.CM_Get_Parent.argtypes = [
        ctypes.POINTER(wintypes.DWORD),
        wintypes.DWORD,
        wintypes.ULONG,
    ]

    def port_name(devices: int, info: SP_DEVINFO_DATA) -> str | None:
        key = setupapi.SetupDiOpenDevRegKey(
            devices, ctypes.byref(info), DICS_FLAG_GLOBAL, 0, DIREG_DEV, KEY_READ
        )
        if key in (None, INVALID_HANDLE):
            return None
        try:
            buf = ctypes.create_unicode_buffer(64)
            size = wintypes.DWORD(ctypes.sizeof(buf))
            if advapi.RegQueryValueExW(
                key, "PortName", None, None, buf, ctypes.byref(size)
            ):
                return None
            return buf.value
        finally:
            advapi.RegCloseKey(key)

    def description(devinst: int) -> str | None:
        buf = ctypes.create_unicode_buffer(256)
        size = wintypes.ULONG(ctypes.sizeof(buf))
        prop_type = wintypes.ULONG()
        if cfgmgr.CM_Get_DevNode_PropertyW(
            devinst,
            ctypes.byref(bus_reported_desc),
            ctypes.byref(prop_type),
            buf,
            ctypes.byref(size),
            0,
        ):
            return None
        return buf.value or None

    devices = setupapi.SetupDiGetClassDevsW(
        ctypes.byref(ports_class), None, None, DIGCF_PRESENT
    )
    if devices in (None, INVALID_HANDLE):
        return None
    try:
        info = SP_DEVINFO_DATA()
        info.cbSize = ctypes.sizeof(SP_DEVINFO_DATA)
        index = 0
        while setupapi.SetupDiEnumDeviceInfo(devices, index, ctypes.byref(info)):
            index += 1
            if port_name(devices, info) != device:
                continue
            # The COM port's own node, then up to the USB device node: the
            # first description that is the board's wins, else the first one.
            first: str | None = None
            devinst = info.DevInst
            for _ in range(3):
                found = description(devinst)
                if parse_product(found)[0]:
                    return found
                first = first or found
                parent = wintypes.DWORD()
                if cfgmgr.CM_Get_Parent(ctypes.byref(parent), devinst, 0):
                    break
                devinst = parent.value
            return first
        return None
    finally:
        setupapi.SetupDiDestroyDeviceInfoList(devices)
