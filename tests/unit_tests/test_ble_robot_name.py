"""The BLE service advertises the user-set robot name.

A robot that was never renamed keeps advertising "ReachyMini". Once renamed
(from the app over WebRTC, or over BLE with SET_NAME), the advert and the
adapter Alias follow, without a restart.

As with the other BLE tests, ``dbus`` is stubbed before importing the module (it
is not a project dependency — the service runs under the robot's system
Python), and the suite is Linux-only.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="BLE provisioning service is Linux-only"
)

_SERVICE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/reachy_mini/daemon/app/services/bluetooth/bluetooth_service.py"
)


def _import_bluetooth_service():
    """Load bluetooth_service with a stubbed ``dbus`` (absent from the venv)."""
    service = types.ModuleType("dbus.service")

    class _Object:
        def __init__(self, *a, **k):
            pass

    def _decorator_factory(*a, **k):
        return lambda fn: fn

    service.Object = _Object
    service.method = _decorator_factory
    service.signal = _decorator_factory

    exceptions = types.ModuleType("dbus.exceptions")

    class DBusException(Exception):
        pass

    exceptions.DBusException = DBusException

    mainloop = types.ModuleType("dbus.mainloop")
    mainloop_glib = types.ModuleType("dbus.mainloop.glib")
    mainloop_glib.DBusGMainLoop = MagicMock()
    mainloop.glib = mainloop_glib

    dbus = types.ModuleType("dbus")
    dbus.service = service
    dbus.exceptions = exceptions
    dbus.mainloop = mainloop
    dbus.__getattr__ = lambda name: MagicMock()

    stubs = {
        "dbus": dbus,
        "dbus.service": service,
        "dbus.exceptions": exceptions,
        "dbus.mainloop": mainloop,
        "dbus.mainloop.glib": mainloop_glib,
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "_ble_robot_name_under_test", _SERVICE_PATH
        )
        bt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bt)
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return bt


if sys.platform == "linux":
    bt = _import_bluetooth_service()


@pytest.fixture
def name_file(tmp_path, monkeypatch):
    """Point the service at a temporary robot_name.json."""
    path = tmp_path / "robot_name.json"
    monkeypatch.setattr(bt, "ROBOT_NAME_PATH", path)
    return path


def _service(connected: bool = False):
    """Build a started-looking service: adapter props and a registered advert."""
    svc = bt.BluetoothCommandService(pin_code="12345")
    svc._adapter_props = MagicMock()
    svc.adv = MagicMock(local_name=bt.DEFAULT_DEVICE_NAME)
    svc._connected_device_path = "/org/bluez/hci0/dev_PHONE" if connected else None
    svc._reassert_advertising = MagicMock()
    return svc


def test_persisted_name_is_read_and_trimmed(name_file):
    """The stored name comes back without surrounding whitespace."""
    name_file.write_text(json.dumps({"name": "  Bob  "}))
    assert bt.get_persisted_robot_name() == "Bob"


@pytest.mark.parametrize(
    "content", [None, "", "not json", "[]", '{"name": "   "}', '{"name": 3}']
)
def test_no_usable_name_means_not_renamed(name_file, content):
    """A missing, corrupt or blank file reads as never renamed."""
    if content is not None:
        name_file.write_text(content)
    assert bt.get_persisted_robot_name() is None


def test_advertised_name_fits_the_scan_response():
    """Short names pass through; long ones are cut to the byte budget."""
    assert bt.advertised_name("Bob") == "Bob"
    assert bt.advertised_name("x" * 64) == "x" * bt.MAX_ADVERTISED_NAME_BYTES


def test_advertised_name_never_splits_a_character():
    """The cut lands on a UTF-8 character boundary."""
    # 2 bytes each: 29 bytes would cut the 15th one in half.
    assert bt.advertised_name("é" * 20) == "é" * 14


def test_never_renamed_robot_keeps_the_default_name(name_file):
    """No stored name: the advert and the adapter Alias are left alone."""
    svc = _service()
    svc._apply_persisted_name()
    assert svc.device_name == bt.DEFAULT_DEVICE_NAME
    svc._adapter_props.Set.assert_not_called()
    svc._reassert_advertising.assert_not_called()


def test_rename_updates_advert_and_alias(name_file):
    """A stored name is advertised and set as the adapter Alias."""
    name_file.write_text(json.dumps({"name": "Bob"}))
    svc = _service()
    svc._apply_persisted_name()
    assert svc.device_name == "Bob"
    assert svc.adv.local_name == "Bob"
    assert svc._adapter_props.Set.call_args.args[:2] == ("org.bluez.Adapter1", "Alias")
    svc._reassert_advertising.assert_called_once()


def test_rename_while_connected_waits_for_the_disconnect(name_file):
    """A connected central is not disturbed; the disconnect re-advertises."""
    name_file.write_text(json.dumps({"name": "Bob"}))
    svc = _service(connected=True)
    svc._apply_persisted_name()
    assert svc.adv.local_name == "Bob"
    svc._reassert_advertising.assert_not_called()


def test_unchanged_name_does_not_readvertise(name_file):
    """Polling the same name re-registers the advert only once."""
    name_file.write_text(json.dumps({"name": "Bob"}))
    svc = _service()
    svc._apply_persisted_name()
    svc._apply_persisted_name()
    assert svc._poll_device_name() is True  # GLib timer keeps firing
    svc._reassert_advertising.assert_called_once()


def test_set_name_applies_the_new_name_right_away(monkeypatch):
    """SET_NAME schedules the advert update instead of waiting for the poll."""
    svc = _service()
    monkeypatch.setattr(svc, "_is_authed", lambda device: True)
    monkeypatch.setattr(svc, "_run_async", lambda fn: fn())
    monkeypatch.setattr(bt, "_set_robot_name", lambda name: f"OK: Named {name}")
    idle_add = MagicMock()
    monkeypatch.setattr(bt.GLib, "idle_add", idle_add)

    assert svc._handle_command(b"SET_NAME Bob", device="/dev") == "OK: working"
    idle_add.assert_called_once_with(svc._apply_persisted_name)
