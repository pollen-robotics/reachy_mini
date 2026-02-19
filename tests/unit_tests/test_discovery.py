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


@patch("reachy_mini.utils.discovery._filter_alive")
@patch("reachy_mini.utils.discovery._find_robots_dnssd")
@patch("reachy_mini.utils.discovery._find_robots_zeroconf")
def test_find_robots_returns_collected(mock_zeroconf, mock_dnssd, mock_alive):
    """find_robots() returns robots from the platform-appropriate backend."""
    robot = DiscoveredRobot(
        name="test", host="test.local.", port=7777, addresses=["10.0.0.1"],
    )

    # Simulate: zeroconf finds nothing, dnssd finds the robot (macOS path),
    # or zeroconf finds the robot directly (Linux path).
    import sys
    if sys.platform == "darwin":
        mock_zeroconf.return_value = []
        mock_dnssd.return_value = [robot]
    else:
        mock_zeroconf.return_value = [robot]

    mock_alive.side_effect = lambda robots: robots

    robots = find_robots(timeout=0.1)
    assert len(robots) == 1
    assert robots[0].name == "test"
    assert robots[0].port == 7777
    mock_alive.assert_called_once()


@patch("reachy_mini.utils.discovery._filter_alive", return_value=[])
@patch("reachy_mini.utils.discovery._find_robots_dnssd", return_value=[])
@patch("reachy_mini.utils.discovery._find_robots_zeroconf", return_value=[])
def test_find_robots_returns_empty(mock_zeroconf, mock_dnssd, mock_alive):
    """find_robots() returns empty list when no robots are found."""
    robots = find_robots(timeout=0.1)
    assert len(robots) == 0


@patch("reachy_mini.utils.discovery.socket.create_connection")
def test_filter_alive_keeps_reachable(mock_conn):
    """_filter_alive keeps robots that accept TCP connections."""
    from reachy_mini.utils.discovery import _filter_alive

    mock_conn.return_value.__enter__ = MagicMock()
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)

    robot = DiscoveredRobot(
        name="test", host="test.local.", port=7777, addresses=["10.0.0.1"],
    )
    result = _filter_alive([robot])
    assert len(result) == 1
    mock_conn.assert_called_once_with(("10.0.0.1", 7777), timeout=0.5)


@patch("reachy_mini.utils.discovery.socket.create_connection", side_effect=OSError)
def test_filter_alive_removes_unreachable(mock_conn):
    """_filter_alive removes robots that can't be reached."""
    from reachy_mini.utils.discovery import _filter_alive

    robot = DiscoveredRobot(
        name="stale", host="stale.local.", port=8000, addresses=["10.0.0.99"],
    )
    result = _filter_alive([robot])
    assert len(result) == 0


def test_service_type_constant():
    """SERVICE_TYPE follows DNS-SD naming convention."""
    assert SERVICE_TYPE == "_reachy-mini._tcp.local."
