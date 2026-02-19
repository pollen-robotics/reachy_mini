"""mDNS service registration and discovery for Reachy Mini robots."""

from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Dict, List

from zeroconf import ServiceBrowser, ServiceInfo, ServiceListener, Zeroconf

logger = logging.getLogger(__name__)

SERVICE_TYPE = "_reachy-mini._tcp.local."


@dataclass
class DiscoveredRobot:
    """A Reachy Mini robot discovered via mDNS."""

    name: str
    host: str
    port: int
    addresses: List[str] = field(default_factory=list)
    properties: Dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return a human-readable representation."""
        return f"DiscoveredRobot(name={self.name!r}, host={self.host!r}, port={self.port})"


class MdnsServiceRegistration:
    """Register a Reachy Mini daemon as an mDNS service."""

    def __init__(self, robot_name: str, port: int) -> None:
        """Initialize with robot name and port to advertise."""
        self._robot_name = robot_name
        self._port = port
        self._zeroconf: Zeroconf | None = None
        self._info: ServiceInfo | None = None
        self._register_thread: threading.Thread | None = None

    def register(self) -> None:
        """Register the mDNS service in a background thread.

        Runs in a separate thread so it's safe to call from an async context.
        Logs warning on failure, never raises.
        """
        self._register_thread = threading.Thread(target=self._do_register, daemon=True)
        self._register_thread.start()

    def unregister(self) -> None:
        """Unregister the mDNS service. No-op if not registered.

        Runs in a separate thread and waits for completion so the service
        is guaranteed to be unregistered before the caller continues.
        """
        # Wait for register to finish first
        if self._register_thread is not None:
            self._register_thread.join(timeout=10.0)
            self._register_thread = None

        if self._zeroconf is None or self._info is None:
            return

        thread = threading.Thread(target=self._do_unregister, daemon=True)
        thread.start()
        thread.join(timeout=5.0)

    def _do_register(self) -> None:
        try:
            pkg_version = version("reachy_mini")
        except Exception:
            pkg_version = "unknown"

        properties = {
            "version": pkg_version,
            "robot_name": self._robot_name,
            "ws_path": "/ws/sdk",
        }

        try:
            self._zeroconf = Zeroconf()
            self._info = ServiceInfo(
                SERVICE_TYPE,
                name=f"{self._robot_name}.{SERVICE_TYPE}",
                port=self._port,
                properties=properties,
                server=f"{socket.gethostname()}.local.",
            )
            self._zeroconf.register_service(self._info)
            logger.info(
                "mDNS service registered: %s on port %d",
                self._robot_name,
                self._port,
            )
        except Exception:
            logger.warning("Failed to register mDNS service", exc_info=True)
            self._close_zeroconf()

    def _do_unregister(self) -> None:
        try:
            assert self._zeroconf is not None and self._info is not None
            self._zeroconf.unregister_service(self._info)
            logger.info("mDNS service unregistered: %s", self._robot_name)
        except Exception:
            logger.warning("Failed to unregister mDNS service", exc_info=True)
        finally:
            self._close_zeroconf()

    def _close_zeroconf(self) -> None:
        if self._zeroconf is not None:
            try:
                self._zeroconf.close()
            except Exception:
                pass
            self._zeroconf = None
            self._info = None


class _RobotCollector(ServiceListener):
    """Listener that collects discovered robots."""

    def __init__(self) -> None:
        self.robots: List[DiscoveredRobot] = []

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Handle a newly discovered service."""
        info = zc.get_service_info(type_, name)
        if info is None:
            return

        addresses = [socket.inet_ntoa(addr) for addr in info.addresses]
        props = {
            k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else str(v)
            for k, v in info.properties.items()
        }

        robot_name = props.get("robot_name", name.removesuffix(f".{SERVICE_TYPE}"))

        self.robots.append(
            DiscoveredRobot(
                name=robot_name,
                host=info.server or "",
                port=info.port or 0,
                addresses=addresses,
                properties=props,
            )
        )

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Handle a removed service."""

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Handle an updated service."""


def find_robots(timeout: float = 3.0) -> List[DiscoveredRobot]:
    """Discover Reachy Mini robots on the local network via mDNS.

    Returns as soon as at least one robot is found, or after ``timeout`` seconds.

    Args:
        timeout: Maximum time to wait for responses, in seconds.

    Returns:
        A list of discovered robots.

    """
    zc = Zeroconf()

    collector = _RobotCollector()
    browser = ServiceBrowser(zc, SERVICE_TYPE, listener=collector)

    try:
        time.sleep(timeout)
    finally:
        browser.cancel()
        zc.close()

    # Filter out stale entries by checking TCP connectivity
    alive: List[DiscoveredRobot] = []
    for robot in collector.robots:
        for addr in robot.addresses:
            try:
                with socket.create_connection((addr, robot.port), timeout=0.5):
                    alive.append(robot)
                    break
            except OSError:
                continue

    return alive


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Discover Reachy Mini robots on the local network.")
    parser.add_argument("--timeout", type=float, default=3.0, help="Discovery timeout in seconds (default: 3.0)")
    args = parser.parse_args()

    robots = find_robots(timeout=args.timeout)

    if not robots:
        print("No robots found.")
    else:
        for robot in robots:
            addrs = ", ".join(robot.addresses)
            print(f"{robot.name} - {addrs}:{robot.port} ({robot.host})")
