"""Unit tests for the WiFi station-connect rescan/retry logic.

``_connect_station_with_rescan`` retries ONLY the transient SSID-not-found race
(``nmcli.NotExistException``) that a rescan fixes; every other failure — notably
a wrong password (``ConnectionActivateFailedException``) — must propagate
immediately so the caller falls back to hotspot without burning seconds.

The module imports ``nmcli`` (a Linux-only dependency that runs under the
daemon venv on the robot), so the suite is skipped off-Linux. Importing the
module also kicks off ``ensure_wifi_on_startup()`` on a real thread that shells
out to ``sudo nmcli``; we neutralise ``threading.Thread`` during import so the
test never touches WiFi hardware.
"""
import importlib.util
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="nmcli is a Linux-only dependency"
)

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/reachy_mini/daemon/app/routers/wifi_config.py"
)


class _NoopThread:
    """Stand-in so importing the module doesn't fire ensure_wifi_on_startup()
    (which shells out to ``sudo nmcli``) as a real background thread."""

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False


def _import_wifi_config():
    real_thread = threading.Thread
    threading.Thread = _NoopThread
    try:
        spec = importlib.util.spec_from_file_location(
            "_wifi_config_under_test", _MODULE_PATH
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        threading.Thread = real_thread
    return mod


# Imported once at module scope; only runs on Linux thanks to the guard.
if sys.platform == "linux":
    wifi = _import_wifi_config()


@pytest.fixture
def sleeps(monkeypatch):
    """Make retry/settle delays instant and record their durations."""
    recorded = []
    monkeypatch.setattr(wifi.time, "sleep", lambda s: recorded.append(s))
    return recorded


@pytest.fixture
def device(monkeypatch):
    """Patchable stand-ins for the two nmcli device calls under test."""
    connect = MagicMock(name="wifi_connect")
    rescan = MagicMock(name="wifi_rescan")
    monkeypatch.setattr(wifi.nmcli.device, "wifi_connect", connect)
    monkeypatch.setattr(wifi.nmcli.device, "wifi_rescan", rescan)
    return connect, rescan


def test_transient_ssid_not_found_then_success(device, sleeps):
    """A few NotExistException misses are retried with a rescan each time."""
    connect, rescan = device
    attempts = {"n": 0}

    def flaky(ssid, password):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise wifi.nmcli.NotExistException("No network with SSID 'X' found")

    connect.side_effect = flaky
    wifi._connect_station_with_rescan(ssid="X", password="pw")  # returns on 3rd
    assert connect.call_count == 3
    assert rescan.call_count == 3


def test_transient_exhausted_reraises(device, sleeps):
    """A persistently-missing SSID re-raises NotExistException after the cap."""
    connect, _ = device
    connect.side_effect = wifi.nmcli.NotExistException("still missing")
    with pytest.raises(wifi.nmcli.NotExistException):
        wifi._connect_station_with_rescan(ssid="X", password="pw")
    assert connect.call_count == wifi.WIFI_CONNECT_MAX_RETRIES


def test_wrong_password_fails_fast(device, sleeps):
    """A terminal auth failure is NOT retried — propagates on the first try."""
    connect, _ = device
    connect.side_effect = wifi.nmcli.ConnectionActivateFailedException(
        "Connection activation failed"
    )
    with pytest.raises(wifi.nmcli.ConnectionActivateFailedException):
        wifi._connect_station_with_rescan(ssid="X", password="wrong")
    assert connect.call_count == 1
    # No inter-attempt retry delay should have fired.
    assert wifi.WIFI_CONNECT_RETRY_DELAY not in sleeps


def test_rescan_failure_is_non_fatal(device, sleeps):
    """A failed rescan (e.g. one already in flight) doesn't abort the connect."""
    connect, rescan = device
    rescan.side_effect = Exception("Scanning not allowed")
    wifi._connect_station_with_rescan(ssid="X", password="pw")  # still succeeds
    assert connect.call_count == 1
