"""Unit tests for the wrong-PIN throttle on the BLE WiFi-provisioning session.

These exercise pure throttle logic in ``BluetoothCommandService`` (time math +
PIN compare); no real BLE stack is involved.

The service only runs on Linux, so the suite is skipped elsewhere. Even on
Linux, ``dbus`` (dbus-python) is not a project dependency — the service runs
under the robot's system Python — so we stub it before importing the module.
``gi``/PyGObject *is* a Linux dependency and is left real.
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

# The service ships as a standalone script run under the robot's system Python
# (see install_service_bluetooth.sh); it is never imported via the package
# path and pulls in no reachy_mini modules. Load it directly by file path so we
# don't drag in the whole daemon app package.
_SERVICE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/reachy_mini/daemon/app/services/bluetooth/bluetooth_service.py"
)


def _import_bluetooth_service():
    """Load bluetooth_service with a stubbed ``dbus`` (absent from the venv).

    The GATT classes subclass ``dbus.service.Object`` and use
    ``@dbus.service.method`` at import time, so the stub must provide a real
    base class and a no-op decorator factory — not bare MagicMocks. The
    throttle paths under test never touch dbus at runtime.
    """
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
    dbus.__getattr__ = lambda name: MagicMock()  # any other dbus.X at import

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
            "_ble_bluetooth_service_under_test", _SERVICE_PATH
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


# Imported once at module scope; only runs on Linux thanks to the guard.
if sys.platform == "linux":
    bt = _import_bluetooth_service()


@pytest.fixture
def clock(monkeypatch):
    """Controllable monotonic clock for the bluetooth_service module."""
    state = {"now": 1000.0}
    monkeypatch.setattr(bt.time, "monotonic", lambda: state["now"])
    return state


@pytest.fixture
def svc():
    return bt.BluetoothCommandService(pin_code="12345")


def _pin(svc, value):
    return svc._handle_command(f"PIN_{value}".encode())


def _trip_first_lockout(svc):
    """Send enough wrong PINs to arm the first lockout window."""
    for _ in range(svc.PIN_FREE_ATTEMPTS + 1):
        _pin(svc, "00000")


def test_correct_pin_connects(svc, clock):
    assert _pin(svc, "12345").startswith("OK")
    assert svc.connected is True
    assert svc._authed_until == clock["now"] + svc.SESSION_TTL_S


def test_correct_pin_rejected_during_lockout(svc, clock):
    """The core invariant: a correct PIN landing mid-lockout must not win."""
    _trip_first_lockout(svc)
    assert svc._pin_lockout_remaining() > 0
    resp = _pin(svc, "12345")  # correct PIN, but locked out
    assert "Too many attempts" in resp
    assert svc.connected is False


def test_spam_during_lockout_does_not_extend(svc, clock):
    """Hammering while locked must not escalate the window (no clock advance)."""
    _trip_first_lockout(svc)
    locked_until = svc._pin_locked_until
    for _ in range(20):
        _pin(svc, "00000")
    assert svc._pin_locked_until == locked_until


def test_lockout_expires(svc, clock):
    _trip_first_lockout(svc)
    clock["now"] += svc.PIN_LOCKOUT_BASE_S + 1
    assert svc._pin_lockout_remaining() == 0.0
    # Comparison happens again once the window passes.
    assert "Incorrect PIN" in _pin(svc, "00000")


def test_success_resets_throttle(svc, clock):
    # Several misses, each after the prior lockout expires, then a correct PIN.
    for _ in range(svc.PIN_FREE_ATTEMPTS + 2):
        clock["now"] += 10_000
        _pin(svc, "00000")
    clock["now"] += 10_000
    assert _pin(svc, "12345").startswith("OK")
    assert svc._pin_failures == 0
    assert svc._pin_locked_until == 0.0


def test_disconnect_preserves_throttle(svc, clock, monkeypatch):
    """Reconnecting must not wipe an in-progress lockout."""
    monkeypatch.setattr(svc, "_reassert_advertising", lambda: None)
    _trip_first_lockout(svc)
    failures, locked = svc._pin_failures, svc._pin_locked_until
    svc._on_central_disconnected()
    assert svc._pin_failures == failures
    assert svc._pin_locked_until == locked
