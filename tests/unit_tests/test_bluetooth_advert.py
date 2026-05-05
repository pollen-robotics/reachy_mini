"""Unit tests for the BLE advertisement TLV encoder.

Targets the pure-Python ``encode_advert_payload`` helper. The full BLE
service is not exercised here (its dbus / glib bindings are
integration-test territory); we only lock the wire shape so a future
refactor cannot accidentally break the mobile picker's parser.

The encoder lives in the system-Python ``bluetooth_service.py`` module
that runs outside the daemon's venv. We import it directly with
``importlib`` after stubbing the dbus / gi modules: their attributes
on the helper paths we exercise here are reduced to ``UInt16``, ``Byte``
and ``Array`` plumbing.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "reachy_mini"
    / "daemon"
    / "app"
    / "services"
    / "bluetooth"
    / "bluetooth_service.py"
)


def _stub_bus_modules() -> None:
    """Install minimal stubs for ``dbus`` / ``gi`` so the module imports.

    The encoder we test only touches ``dbus.UInt16``, ``dbus.Byte`` and
    ``dbus.Array``; everything else used by the BLE service top-level
    (Object, mainloop bindings, ...) is exposed as a no-op placeholder
    so the import side-effects don't blow up.
    """
    stubs: dict[str, list[str]] = {
        "dbus": ["service", "mainloop"],
        "dbus.mainloop": ["glib"],
        "dbus.service": [],
        "dbus.mainloop.glib": [],
        "dbus.exceptions": [],
        "gi": [],
        "gi.repository": ["GLib"],
    }
    for name, attrs in stubs.items():
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
        mod = sys.modules[name]
        for attr in attrs:
            full = f"{name}.{attr}"
            if not hasattr(mod, attr):
                child = types.ModuleType(full)
                setattr(mod, attr, child)
                sys.modules[full] = child

    # Plumbing the encoder actually exercises.
    sys.modules["dbus"].UInt16 = lambda x: int(x)
    sys.modules["dbus"].Byte = lambda x: int(x) & 0xFF
    sys.modules["dbus"].Array = lambda items, signature=None: list(items)
    # Filler so other module-top-level statements don't crash.
    sys.modules["dbus"].SystemBus = type("SystemBus", (), {})
    sys.modules["dbus"].Interface = type("Interface", (), {})
    sys.modules["dbus"].Dictionary = dict
    sys.modules["dbus"].Boolean = lambda x: bool(x)
    sys.modules["dbus"].String = str
    sys.modules["dbus"].ObjectPath = str
    sys.modules["dbus.service"].Object = type("Object", (), {})
    sys.modules["dbus.service"].method = lambda *a, **k: (lambda f: f)
    sys.modules["dbus.service"].signal = lambda *a, **k: (lambda f: f)
    sys.modules["dbus.service"].BusName = type(
        "BusName", (), {"__init__": lambda self, *a, **k: None}
    )
    sys.modules["dbus.exceptions"].DBusException = type(
        "DBusException", (Exception,), {}
    )
    sys.modules["gi.repository"].GLib = type(
        "GLib", (), {"MainLoop": lambda: None, "timeout_add_seconds": lambda *a: None}
    )


def _import_bt_service() -> types.ModuleType:
    _stub_bus_modules()
    spec = importlib.util.spec_from_file_location("bt_advert_test", _MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bt():  # type: ignore[no-untyped-def]
    return _import_bt_service()


# ---------------------------------------------------------------------------
# encode_advert_payload
# ---------------------------------------------------------------------------


def _decode_payload(bt, mfg_data: dict) -> list[int]:
    """Pull the raw byte list out of the dbus-style ManufacturerData dict."""
    if not mfg_data:
        return []
    assert list(mfg_data.keys()) == [bt.POLLEN_MANUFACTURER_ID]
    return list(mfg_data[bt.POLLEN_MANUFACTURER_ID])


def test_encode_emits_version_byte_then_hwid_tlv(bt) -> None:
    """A canonical 16-hex-char hwid produces a 11-byte payload:
    1 version byte + 1 tag + 1 len + 8 hwid bytes."""
    payload = _decode_payload(bt, bt.encode_advert_payload("0123456789abcdef"))
    assert payload == [
        bt._BLE_ADVERT_FORMAT_VERSION,
        bt._TLV_TAG_HARDWARE_ID_PREFIX,
        bt._TLV_HARDWARE_ID_PREFIX_LEN,
        0x01,
        0x23,
        0x45,
        0x67,
        0x89,
        0xAB,
        0xCD,
        0xEF,
    ]


def test_encode_returns_empty_when_hwid_is_none(bt) -> None:
    """No ``hardware_id`` -> no advertisement payload at all (the
    advert still registers with just Flags + LocalName)."""
    assert bt.encode_advert_payload(None) == {}


def test_encode_returns_empty_on_short_hwid(bt) -> None:
    """A truncated hex string is treated as missing rather than
    silently advertising padding bytes."""
    assert bt.encode_advert_payload("dead") == {}


def test_encode_returns_empty_on_non_hex(bt) -> None:
    """A non-hex string never reaches the wire as garbage bytes."""
    assert bt.encode_advert_payload("not-a-hex-string-zzzzzzz") == {}


def test_encode_uses_only_first_8_bytes_of_hwid(bt) -> None:
    """``hardware_id`` is 16 hex chars by spec, but if a longer string
    arrives we MUST take exactly the first 8 bytes (16 hex chars) and
    drop the rest. This guards the budget: any extra byte would push
    the advert past 31 bytes total once Flags + LocalName is counted."""
    long_hwid = "0123456789abcdef" + "ff" * 16
    payload = _decode_payload(bt, bt.encode_advert_payload(long_hwid))
    assert payload[3:] == [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF]


def test_encode_payload_total_size_under_budget(bt) -> None:
    """Sanity bound: the manufacturer-specific bytes must fit inside
    the 16-byte slot we reserved when sizing the advert (31 bytes -
    Flags 3 - LocalName 12 = 16). A regression that pushes the payload
    over this size will cause BlueZ to silently drop the advert."""
    payload = _decode_payload(bt, bt.encode_advert_payload("a" * 16))
    assert len(payload) <= 16


def test_format_version_is_v0x02(bt) -> None:
    """Wire-shape lock: any change here breaks every parser in the
    fleet at once. Bump only with a coordinated client release."""
    assert bt._BLE_ADVERT_FORMAT_VERSION == 0x02


def test_hwid_tlv_tag_and_len_are_stable(bt) -> None:
    """Same lock as the format version: the parser keys off these
    constants byte-for-byte."""
    assert bt._TLV_TAG_HARDWARE_ID_PREFIX == 0x01
    assert bt._TLV_HARDWARE_ID_PREFIX_LEN == 8
