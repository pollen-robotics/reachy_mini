"""Unit tests for the daemon-update-over-BLE proxy helpers.

These exercise the pure relay/parse/error-mapping logic of the ``_update_*``
functions in ``bluetooth_service`` with ``_daemon_request`` mocked out — no real
BLE stack or daemon is involved.

The service only runs on Linux, so the suite is skipped elsewhere. Even on
Linux, ``dbus`` (dbus-python) is not a project dependency — the service runs
under the robot's system Python — so we stub it before importing the module.
``gi``/PyGObject *is* a Linux dependency and is left real.
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
            "_ble_update_proxy_under_test", _SERVICE_PATH
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


# --- UPDATE_CHECK -----------------------------------------------------------


def test_update_check_available(monkeypatch):
    _patch_daemon(
        monkeypatch,
        {
            "update": {
                "reachy_mini": {
                    "is_available": True,
                    "current_version": "1.8.1",
                    "available_version": "1.9.0",
                }
            }
        },
    )
    out = json.loads(bt._update_check())
    assert out == {"available": True, "current": "1.8.1", "latest": "1.9.0"}


def test_update_check_busy_maps_to_in_progress(monkeypatch):
    _patch_daemon(monkeypatch, _http_error(400))
    assert bt._update_check() == "ERROR: Update in progress"


def test_update_check_daemon_unreachable(monkeypatch):
    _patch_daemon(monkeypatch, bt.urllib.error.URLError("boom"))
    assert bt._update_check() == "ERROR: Daemon unreachable"


# --- UPDATE_START -----------------------------------------------------------


def test_update_start_returns_job_id(monkeypatch):
    _patch_daemon(monkeypatch, {"job_id": "abc-123"})
    assert bt._update_start() == "OK: Update started abc-123"


def test_update_start_missing_job_id(monkeypatch):
    _patch_daemon(monkeypatch, {})
    assert bt._update_start() == "ERROR: Start failed"


def test_update_start_400_maps_to_no_update(monkeypatch):
    _patch_daemon(monkeypatch, _http_error(400))
    assert bt._update_start() == "ERROR: No update available or already in progress"


# --- UPDATE_INFO ------------------------------------------------------------


def test_update_info_requires_job_id():
    assert bt._update_info("   ") == "ERROR: Missing job_id"


def test_update_info_non_dict_is_unknown_job(monkeypatch):
    _patch_daemon(monkeypatch, None)
    assert bt._update_info("job") == "ERROR: Unknown job"


def test_update_info_404_is_unknown_job(monkeypatch):
    _patch_daemon(monkeypatch, _http_error(404))
    assert bt._update_info("job") == "ERROR: Unknown job"


def test_update_info_small_payload_preserves_last(monkeypatch):
    _patch_daemon(monkeypatch, {"status": "in_progress", "logs": ["downloading"]})
    out = json.loads(bt._update_info("job"))
    assert out == {"status": "in_progress", "lines": 1, "last": "downloading"}


def test_update_info_bounds_payload_to_mtu(monkeypatch):
    """The core fix: a huge log line must not overflow the notification MTU."""
    huge = "x" * 5000
    _patch_daemon(monkeypatch, {"status": "in_progress", "logs": ["a", "b", huge]})
    raw = bt._update_info("job")
    # Stays within the byte budget and remains valid JSON the client can parse.
    assert len(raw.encode("utf-8")) <= bt._UPDATE_MTU_BUDGET
    out = json.loads(raw)
    assert out["status"] == "in_progress"
    assert out["lines"] == 3
    # `last` is a (trimmed) prefix of the original final line, never garbage.
    assert out["last"] and huge.startswith(out["last"])
