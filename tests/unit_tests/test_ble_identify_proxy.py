"""Unit tests for the identify-over-BLE proxy helper.

These exercise the pure relay/error-mapping logic of ``_identify`` in
``bluetooth_service`` with ``_daemon_request`` mocked out — no real BLE stack
or daemon is involved.

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
            "_ble_identify_proxy_under_test", _SERVICE_PATH
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


def _http_error(code: int):
    return bt.urllib.error.HTTPError(
        url="http://127.0.0.1:8000", code=code, msg="x", hdrs=None, fp=None
    )


def _patch_daemon(monkeypatch, result):
    """Make ``_daemon_request`` return ``result`` (or raise it, if an Exception)."""

    def fake(*a, **k):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(bt, "_daemon_request", fake)


# --- IDENTIFY ---------------------------------------------------------------


def test_identify_success(monkeypatch):
    _patch_daemon(monkeypatch, {"status": "ok"})
    assert bt._identify() == "OK: Playing identification sound"


def test_identify_posts_builtin_sound_to_media_route(monkeypatch):
    """The proxy must POST the built-in asset to the /api-prefixed media route."""
    captured: dict = {}

    def fake(method, path, params=None, data=None, timeout=None):
        captured["method"] = method
        captured["path"] = path
        captured["data"] = data
        return {"status": "ok"}

    monkeypatch.setattr(bt, "_daemon_request", fake)
    bt._identify()
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/media/play_sound"
    assert captured["data"] == {"file": bt._IDENTIFY_SOUND_FILE}


def test_identify_503_maps_to_audio_not_ready(monkeypatch):
    _patch_daemon(monkeypatch, _http_error(503))
    assert bt._identify() == "ERROR: Audio not ready"


def test_identify_other_http_error_maps_to_identify_failed(monkeypatch):
    _patch_daemon(monkeypatch, _http_error(500))
    assert bt._identify() == "ERROR: Identify failed"


def test_identify_daemon_unreachable(monkeypatch):
    _patch_daemon(monkeypatch, bt.urllib.error.URLError("boom"))
    assert bt._identify() == "ERROR: Daemon unreachable"


def test_identify_unexpected_error_is_reported(monkeypatch):
    _patch_daemon(monkeypatch, RuntimeError("kaboom"))
    assert bt._identify() == "ERROR: kaboom"
