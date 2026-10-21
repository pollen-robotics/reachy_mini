"""Regression tests for the BLE auth-bypass race (GHSA-993g-hgjh-whmf).

Authentication must be bound to the central that entered the PIN: the BlueZ
``options["device"]`` object path. A second nearby central must not inherit the
first's authenticated session, even by racing a request in before the one-shot
``connected`` flag is reset.

These exercise the pure auth-gate logic in ``BluetoothCommandService``; no real
BLE stack is involved. As with the other BLE tests, ``dbus`` is stubbed before
importing the module (it is not a project dependency — the service runs under
the robot's system Python), and the suite is Linux-only.
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
            "_ble_auth_binding_under_test", _SERVICE_PATH
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

# BlueZ-style Device1 paths — the per-connection identities BlueZ passes in
# options["device"] (different MAC -> different path -> distinguishable clients).
USER_A = "/org/bluez/hci0/dev_AA_AA_AA_AA_AA_AA"
USER_B = "/org/bluez/hci0/dev_BB_BB_BB_BB_BB_BB"


@pytest.fixture
def svc():
    return bt.BluetoothCommandService(pin_code="12345")


def test_pin_binds_session_to_calling_device(svc):
    assert svc._handle_command(b"PIN_12345", USER_A).startswith("OK")
    assert svc._is_authed(USER_A) is True
    # A different central never authenticated — must not inherit A's session.
    assert svc._is_authed(USER_B) is False
    # Nor an unidentifiable caller (no device path -> fail closed).
    assert svc._is_authed(None) is False


def test_user_b_cannot_ride_user_a_cmd_session(svc):
    """The advisory's race: A authenticates, B fires CMD_ from another central."""
    assert svc._handle_command(b"PIN_12345", USER_A).startswith("OK")
    reply = svc._handle_command(b"CMD_whatever", USER_B)
    assert "Not connected" in reply


def test_authed_device_wifi_command_gate(svc):
    """Privileged WiFi commands are gated on the authenticated device."""
    svc._handle_command(b"PIN_12345", USER_A)
    assert "Not connected" in svc._handle_command(b"WIFI_SCAN", USER_B)
    # A itself passes the gate (dispatches async work, acked immediately).
    assert svc._handle_command(b"WIFI_SCAN", USER_A).startswith("OK")


def test_disconnect_clears_authed_device(svc, monkeypatch):
    monkeypatch.setattr(svc, "_reassert_advertising", lambda: None)
    svc._handle_command(b"PIN_12345", USER_A)
    svc._on_central_disconnected()
    assert svc._authed_device is None
    assert svc._is_authed(USER_A) is False


def _connected(svc, path, state):
    """Drive the BlueZ Device1 Connected property change for `path`."""
    svc._on_device_properties_changed(
        "org.bluez.Device1", {"Connected": state}, [], path
    )


def test_authed_disconnect_clears_even_when_another_central_is_tracked(
    svc, monkeypatch
):
    """A's session must not outlive A just because B connected after it.

    `_connected_device_path` tracks whichever central connected most recently,
    so an attacker connecting as B makes A's later disconnect skip cleanup. A's
    session would then stay live for the rest of SESSION_TTL_S with A gone, and
    a reconnect spoofing A's address (the Device1 path is MAC-derived) would
    inherit it.
    """
    monkeypatch.setattr(svc, "_reassert_advertising", lambda: None)
    svc._handle_command(b"PIN_12345", USER_A)
    assert svc._is_authed(USER_A) is True

    # B connects and becomes the tracked central.
    _connected(svc, USER_B, True)
    assert svc._connected_device_path == USER_B

    # A — the authenticated central — drops.
    _connected(svc, USER_A, False)

    assert svc._authed_device is None
    assert svc._is_authed(USER_A) is False
    # B is still on the link, so it stays tracked.
    assert svc._connected_device_path == USER_B
