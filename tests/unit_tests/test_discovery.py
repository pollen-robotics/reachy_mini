"""Tests for mDNS service registration and discovery."""

from unittest.mock import MagicMock, patch

import pytest

zeroconf = pytest.importorskip("zeroconf")

from zeroconf import ServiceInfo, Zeroconf

from reachy_mini.utils.discovery import (
    SERVICE_TYPE,
    DiscoveredRobot,
    MdnsServiceRegistration,
    _RobotCollector,
    find_robots,
)


def test_discovered_robot_dataclass():
    """DiscoveredRobot stores name, host, port, addresses, and properties."""
    robot = DiscoveredRobot(
        name="test_robot",
        host="test.local.",
        port=8000,
        addresses=["192.168.1.10"],
        properties={"version": "1.0.0", "robot_name": "test_robot"},
    )
    assert robot.name == "test_robot"
    assert robot.host == "test.local."
    assert robot.port == 8000
    assert robot.addresses == ["192.168.1.10"]
    assert robot.properties["version"] == "1.0.0"


def test_discovered_robot_defaults():
    """DiscoveredRobot has sensible defaults for addresses and properties."""
    robot = DiscoveredRobot(name="r", host="h", port=1)
    assert robot.addresses == []
    assert robot.properties == {}


def test_register_unregister_lifecycle():
    """MdnsServiceRegistration can register and unregister without error."""
    reg = MdnsServiceRegistration("test_robot", 9999)
    reg.register()
    # Wait for background registration thread to complete
    assert reg._register_thread is not None
    reg._register_thread.join(timeout=10.0)
    try:
        assert reg._zeroconf is not None
        assert reg._info is not None
    finally:
        reg.unregister()
    assert reg._zeroconf is None
    assert reg._info is None


def test_unregister_is_noop_when_not_registered():
    """Calling unregister before register does nothing."""
    reg = MdnsServiceRegistration("test_robot", 9999)
    reg.unregister()  # should not raise


def test_robot_collector_add_service():
    """_RobotCollector correctly parses a ServiceInfo into a DiscoveredRobot."""
    import socket

    collector = _RobotCollector()

    info = ServiceInfo(
        SERVICE_TYPE,
        name=f"my_robot.{SERVICE_TYPE}",
        port=8000,
        properties={b"version": b"1.0.0", b"robot_name": b"my_robot", b"ws_path": b"/ws/sdk"},
        server="my_robot.local.",
        addresses=[socket.inet_aton("192.168.1.42")],
    )

    mock_zc = MagicMock(spec=Zeroconf)
    mock_zc.get_service_info.return_value = info

    collector.add_service(mock_zc, SERVICE_TYPE, f"my_robot.{SERVICE_TYPE}")

    assert len(collector.robots) == 1
    robot = collector.robots[0]
    assert robot.name == "my_robot"
    assert robot.host == "my_robot.local."
    assert robot.port == 8000
    assert robot.addresses == ["192.168.1.42"]
    assert robot.properties["version"] == "1.0.0"
    assert robot.properties["ws_path"] == "/ws/sdk"


def test_robot_collector_skips_none_info():
    """_RobotCollector ignores services that can't be resolved."""
    collector = _RobotCollector()
    mock_zc = MagicMock(spec=Zeroconf)
    mock_zc.get_service_info.return_value = None

    collector.add_service(mock_zc, SERVICE_TYPE, "unknown._reachy-mini._tcp.local.")
    assert len(collector.robots) == 0


@patch("reachy_mini.utils.discovery.socket.create_connection")
@patch("reachy_mini.utils.discovery.ServiceBrowser")
@patch("reachy_mini.utils.discovery.time.sleep")
def test_find_robots_returns_collected(mock_sleep, mock_browser_cls, mock_conn):
    """find_robots() returns robots collected by the listener and verified alive."""
    import socket as sock

    info = ServiceInfo(
        SERVICE_TYPE,
        name=f"test.{SERVICE_TYPE}",
        port=7777,
        properties={b"robot_name": b"test", b"version": b"1.0"},
        server="test.local.",
        addresses=[sock.inet_aton("10.0.0.1")],
    )

    def fake_browser(zc, stype, listener):
        # Simulate discovery by calling the listener
        mock_zc = MagicMock(spec=Zeroconf)
        mock_zc.get_service_info.return_value = info
        listener.add_service(mock_zc, stype, f"test.{stype}")
        browser = MagicMock()
        return browser

    mock_browser_cls.side_effect = fake_browser
    mock_conn.return_value.__enter__ = MagicMock()
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)

    robots = find_robots(timeout=0.1)
    assert len(robots) == 1
    assert robots[0].name == "test"
    assert robots[0].port == 7777
    assert robots[0].addresses == ["10.0.0.1"]
    mock_conn.assert_called_once_with(("10.0.0.1", 7777), timeout=0.5)


@patch("reachy_mini.utils.discovery.socket.create_connection", side_effect=OSError)
@patch("reachy_mini.utils.discovery.ServiceBrowser")
@patch("reachy_mini.utils.discovery.time.sleep")
def test_find_robots_filters_unreachable(mock_sleep, mock_browser_cls, mock_conn):
    """find_robots() filters out robots that fail the liveness check."""
    import socket as sock

    info = ServiceInfo(
        SERVICE_TYPE,
        name=f"stale.{SERVICE_TYPE}",
        port=8000,
        properties={b"robot_name": b"stale", b"version": b"1.0"},
        server="stale.local.",
        addresses=[sock.inet_aton("10.0.0.99")],
    )

    def fake_browser(zc, stype, listener):
        mock_zc = MagicMock(spec=Zeroconf)
        mock_zc.get_service_info.return_value = info
        listener.add_service(mock_zc, stype, f"stale.{stype}")
        return MagicMock()

    mock_browser_cls.side_effect = fake_browser

    robots = find_robots(timeout=0.1)
    assert len(robots) == 0


def test_service_type_constant():
    """SERVICE_TYPE follows DNS-SD naming convention."""
    assert SERVICE_TYPE == "_reachy-mini._tcp.local."
