"""Unit tests for the boot-time WiFi association grace period.

``ensure_wifi_on_startup`` must never seize wlan0 with the hotspot while
NetworkManager is still associating with a known station profile. Since the
daemon service is ordered after ``NetworkManager.service`` (not
``network-online.target``), a DISCONNECTED snapshot right after boot is
expected — the code has to poll for the association before falling back.

The module imports ``nmcli`` (Linux-only); off-Linux the suite is skipped.
Import machinery mirrors ``test_wifi_connect_retry.py``.
"""

import importlib.util
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
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
    """Stand-in so importing the module doesn't fire ensure_wifi_on_startup()."""

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
            "_wifi_config_grace_under_test", _MODULE_PATH
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        threading.Thread = real_thread
    return mod


if sys.platform == "linux":
    wifi = _import_wifi_config()


def _conn(name, device="--"):
    return SimpleNamespace(name=name, device=device)


@pytest.fixture
def harness(monkeypatch):
    """Instant sleeps + controllable clock, scan, mode, and connections."""
    state = {
        "now": 0.0,
        "modes": [],  # consumed by get_current_wifi_mode, last value sticks
        "connections": [],
    }

    def fake_mode():
        if len(state["modes"]) > 1:
            return state["modes"].pop(0)
        return state["modes"][0]

    monkeypatch.setattr(wifi.time, "sleep", lambda s: state.update(now=state["now"] + s))
    monkeypatch.setattr(wifi.time, "monotonic", lambda: state["now"])
    monkeypatch.setattr(wifi, "scan_available_wifi", MagicMock())
    monkeypatch.setattr(wifi, "get_current_wifi_mode", fake_mode)
    monkeypatch.setattr(wifi, "get_wifi_connections", lambda: state["connections"])
    setup = MagicMock(name="setup_wifi_connection")
    monkeypatch.setattr(wifi, "setup_wifi_connection", setup)
    return state, setup


def test_association_in_flight_avoids_hotspot(harness):
    """Known profile associating a few seconds in → no hotspot."""
    state, setup = harness
    state["connections"] = [_conn("HomeWifi")]
    state["modes"] = [
        wifi.WifiMode.DISCONNECTED,  # first snapshot in ensure_wifi_on_startup
        wifi.WifiMode.DISCONNECTED,  # first poll
        wifi.WifiMode.DISCONNECTED,
        wifi.WifiMode.WLAN,  # association completes
    ]
    wifi.ensure_wifi_on_startup()
    setup.assert_not_called()


def test_no_known_networks_starts_hotspot_immediately(harness):
    """Fresh robot with no station profiles → hotspot without any polling."""
    state, setup = harness
    state["connections"] = [_conn("Hotspot")]
    state["modes"] = [wifi.WifiMode.DISCONNECTED]
    wifi.ensure_wifi_on_startup()
    setup.assert_called_once()
    assert state["now"] == 0.0  # no grace wait


def test_association_never_completes_falls_back_to_hotspot(harness):
    """Known profile but its network is absent → hotspot after the grace."""
    state, setup = harness
    state["connections"] = [_conn("HomeWifi")]
    state["modes"] = [wifi.WifiMode.DISCONNECTED]
    wifi.ensure_wifi_on_startup()
    setup.assert_called_once()
    assert state["now"] >= wifi.WIFI_ASSOCIATION_GRACE


def test_already_connected_touches_nothing(harness):
    """Normal boot where NM associated before the daemon got here."""
    state, setup = harness
    state["connections"] = [_conn("HomeWifi", device="wlan0")]
    state["modes"] = [wifi.WifiMode.WLAN]
    wifi.ensure_wifi_on_startup()
    setup.assert_not_called()
    assert state["now"] == 0.0
