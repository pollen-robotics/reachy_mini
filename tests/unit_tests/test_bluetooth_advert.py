"""Unit tests for the BLE advertisement encoder.

Targets the pure-Python ``encode_advert_payload`` helper. The full BLE
service is not exercised here (its dbus / glib bindings are integration
territory); we only lock the wire shape so a future refactor cannot
silently break the mobile picker's parser or the legacy IP-list parsers
that hypothetical older clients may still run.

The encoder lives in the system-Python ``bluetooth_service.py`` module
that runs outside the daemon's venv. We import it directly with
``importlib`` after stubbing the dbus / gi modules: their attributes
on the helper paths we exercise here are reduced to ``UInt16`` /
``Byte`` / ``Array`` plumbing.
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

    sys.modules["dbus"].UInt16 = lambda x: int(x)
    sys.modules["dbus"].Byte = lambda x: int(x) & 0xFF
    sys.modules["dbus"].Array = lambda items, signature=None: list(items)
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


def _decode(bt, mfg_data: dict) -> list[int]:
    """Pull the raw byte list out of the dbus-style ManufacturerData dict."""
    if not mfg_data:
        return []
    assert list(mfg_data.keys()) == [bt.POLLEN_MANUFACTURER_ID]
    return list(mfg_data[bt.POLLEN_MANUFACTURER_ID])


# ---------------------------------------------------------------------------
# Wire-shape locks (the contract any reader must rely on)
# ---------------------------------------------------------------------------


def test_constants_match_the_documented_wire_shape(bt) -> None:
    """The wire is documented as flag(1) + IPv4(4) + hwid_prefix(7).
    These three constants are what the parser keys off; if any of them
    drifts, every BLE-aware client must re-release. A test failing here
    is a wire-shape break, not just a refactor.
    """
    assert bt._HARDWARE_ID_PREFIX_LEN == 7
    assert bt._FLAG_LAN == 0x00
    assert bt._FLAG_HOTSPOT == 0x01


def test_total_payload_size_at_or_below_budget(bt) -> None:
    """Hard cap: 12 bytes for the ManufacturerData payload (after AD
    overhead is taken). 1 (flag) + 4 (IPv4) + 7 (hwid prefix) = 12.
    A regression that pushes past 12 will cause BlueZ to silently drop
    the advert or truncate it, both invisible until the mobile picker
    breaks in the field.
    """
    payload = _decode(
        bt,
        bt.encode_advert_payload(
            "0123456789abcdef" + "00" * 16,  # canonical 16-hex hwid
            "CONNECTED [wlan0] 192.168.1.19",
        ),
    )
    assert len(payload) == 12


# ---------------------------------------------------------------------------
# IP slot
# ---------------------------------------------------------------------------


def test_lan_ip_emitted_with_flag_zero(bt) -> None:
    payload = _decode(
        bt,
        bt.encode_advert_payload(None, "CONNECTED [wlan0] 192.168.1.19"),
    )
    # No hwid -> just flag + IP
    assert payload == [bt._FLAG_LAN, 192, 168, 1, 19]


def test_hotspot_ip_emitted_with_flag_one(bt) -> None:
    """Hotspot is the one case where the daemon's wlan0 hosts an AP at
    10.42.0.1. The flag MUST be 0x01 so older parsers can branch on
    "this robot is in setup mode, not on the LAN"."""
    payload = _decode(bt, bt.encode_advert_payload(None, "HOTSPOT [wlan0] 10.42.0.1"))
    assert payload == [bt._FLAG_HOTSPOT, 10, 42, 0, 1]


def test_offline_with_no_hwid_yields_empty_payload(bt) -> None:
    """Nothing to publish at all -> drop the manufacturer data entry.
    The advertisement still registers with just Flags + LocalName."""
    assert bt.encode_advert_payload(None, "OFFLINE") == {}
    assert bt.encode_advert_payload(None, "ERROR") == {}
    assert bt.encode_advert_payload(None, "") == {}
    assert bt.encode_advert_payload(None, None) == {}


def test_offline_but_hwid_present_emits_hwid_only(bt) -> None:
    """A daemon that boots before the network is up should still
    advertise its hardware id - the mobile picker can't dedupe by IP
    here but can still match the BLE row to the same robot's central
    listing once it gets online."""
    payload = _decode(
        bt, bt.encode_advert_payload("a" * 16, "OFFLINE")
    )
    assert payload == [0xAA] * bt._HARDWARE_ID_PREFIX_LEN


def test_unparseable_network_status_falls_through_to_hwid_only(bt) -> None:
    """A garbled NETWORK_STATUS payload (e.g. daemon crashed mid-write)
    must not break the advertisement: emit hwid only, skip the IP."""
    payload = _decode(
        bt, bt.encode_advert_payload("0123456789abcdef" + "00" * 16, "totally garbled")
    )
    assert len(payload) == bt._HARDWARE_ID_PREFIX_LEN


# ---------------------------------------------------------------------------
# Hardware id slot
# ---------------------------------------------------------------------------


def test_hwid_emitted_at_offset_5(bt) -> None:
    """The mobile parser slices ``payload[5:12]`` to extract hwid. The
    encoder MUST place hwid at offset 5 (right after flag + IPv4)."""
    payload = _decode(
        bt,
        bt.encode_advert_payload(
            "0123456789abcdef" + "00" * 16,
            "CONNECTED [wlan0] 192.168.1.19",
        ),
    )
    # bytes 5..11 = first 7 bytes of hwid hex = 0x01,0x23,0x45,0x67,0x89,0xab,0xcd
    assert payload[5:12] == [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD]


def test_hwid_uses_only_first_7_bytes(bt) -> None:
    """``hardware_id`` is 16 hex chars (= 8 bytes) by spec; we only
    publish the first 7 (= 14 hex chars). The truncation MUST be
    deterministic so the prefix the UI displays (``id:6171b``) is the
    same prefix the parser extracts from the advert."""
    payload = _decode(
        bt,
        bt.encode_advert_payload("0123456789abcdef", "CONNECTED [wlan0] 1.2.3.4"),
    )
    assert payload[5:12] == [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD]
    # The 8th byte of hwid (0xef) is intentionally NOT in the advert.
    assert 0xEF not in payload[5:12]


def test_hwid_none_emits_payload_with_only_ip(bt) -> None:
    payload = _decode(
        bt, bt.encode_advert_payload(None, "CONNECTED [wlan0] 1.2.3.4")
    )
    assert payload == [bt._FLAG_LAN, 1, 2, 3, 4]


def test_hwid_short_or_invalid_silently_falls_back(bt) -> None:
    """A truncated or non-hex hwid is treated as missing rather than
    advertising padding bytes. Same path as ``hwid is None``."""
    short = bt.encode_advert_payload("dead", "CONNECTED [wlan0] 1.2.3.4")
    not_hex = bt.encode_advert_payload(
        "not-a-hex-string-z", "CONNECTED [wlan0] 1.2.3.4"
    )
    assert _decode(bt, short) == [bt._FLAG_LAN, 1, 2, 3, 4]
    assert _decode(bt, not_hex) == [bt._FLAG_LAN, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Combined cases (the canonical happy path)
# ---------------------------------------------------------------------------


def test_canonical_lan_and_hwid_payload(bt) -> None:
    payload = _decode(
        bt,
        bt.encode_advert_payload(
            "6171bd4b0e35ee9b", "CONNECTED [wlan0] 192.168.1.19"
        ),
    )
    assert payload == [
        bt._FLAG_LAN,
        192,
        168,
        1,
        19,
        0x61,
        0x71,
        0xBD,
        0x4B,
        0x0E,
        0x35,
        0xEE,
    ]


def test_canonical_hotspot_and_hwid_payload(bt) -> None:
    payload = _decode(
        bt,
        bt.encode_advert_payload(
            "6171bd4b0e35ee9b", "HOTSPOT [wlan0] 10.42.0.1"
        ),
    )
    assert payload == [
        bt._FLAG_HOTSPOT,
        10,
        42,
        0,
        1,
        0x61,
        0x71,
        0xBD,
        0x4B,
        0x0E,
        0x35,
        0xEE,
    ]
