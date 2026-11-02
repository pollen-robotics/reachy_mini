"""The BLE service drops the bonds phones made with the robot at startup.

A bond the robot keeps for a phone turns into a stale one as soon as the phone
side changes, and BlueZ then refuses the phone's re-pairing, so every connect
from that phone fails. Gamepads, keyboards and speakers are kept.

As with the other BLE tests, ``dbus`` is stubbed before importing the module (it
is not a project dependency — the service runs under the robot's system
Python), and the suite is Linux-only.
"""

import importlib.util
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
            "_ble_forget_bonds_under_test", _SERVICE_PATH
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

ADAPTER = "/org/bluez/hci0"


def _forget(monkeypatch, objects, remove_side_effect=None):
    """Run ``_forget_phone_bonds`` over ``objects``; return the removal attempts."""
    adapter_iface = MagicMock()
    adapter_iface.RemoveDevice.side_effect = remove_side_effect
    object_manager = MagicMock()
    object_manager.GetManagedObjects.return_value = objects

    def interface(obj, name):
        return adapter_iface if name == "org.bluez.Adapter1" else object_manager

    monkeypatch.setattr(bt.dbus, "Interface", interface, raising=False)
    svc = bt.BluetoothCommandService(pin_code="12345")
    svc.bus = MagicMock()
    svc._forget_phone_bonds(MagicMock(object_path=ADAPTER))
    return [c.args[0] for c in adapter_iface.RemoveDevice.call_args_list]


def test_forgets_bonded_phones_only(monkeypatch):
    """Bonded phones go; input/audio devices, unbonded and other adapters stay."""
    device = "org.bluez.Device1"
    removed = _forget(
        monkeypatch,
        {
            ADAPTER: {"org.bluez.Adapter1": {}},
            f"{ADAPTER}/dev_PHONE": {device: {"Paired": True, "Icon": "phone"}},
            f"{ADAPTER}/dev_LE_PHONE": {device: {"Bonded": True}},
            f"{ADAPTER}/dev_PAD": {device: {"Paired": True, "Icon": "input-gaming"}},
            f"{ADAPTER}/dev_SPEAKER": {device: {"Paired": True, "Icon": "audio-card"}},
            f"{ADAPTER}/dev_SEEN": {device: {"Paired": False, "Icon": "phone"}},
            "/org/bluez/hci1/dev_OTHER": {device: {"Paired": True}},
        },
    )
    assert removed == [f"{ADAPTER}/dev_PHONE", f"{ADAPTER}/dev_LE_PHONE"]


def test_a_failing_removal_does_not_stop_the_others(monkeypatch):
    """A device BlueZ refuses to remove doesn't keep the next one bonded."""
    removed = _forget(
        monkeypatch,
        {
            f"{ADAPTER}/dev_A": {"org.bluez.Device1": {"Paired": True}},
            f"{ADAPTER}/dev_B": {"org.bluez.Device1": {"Paired": True}},
        },
        remove_side_effect=[bt.dbus.exceptions.DBusException(), None],
    )
    assert removed == [f"{ADAPTER}/dev_A", f"{ADAPTER}/dev_B"]
