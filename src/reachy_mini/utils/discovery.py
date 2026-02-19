"""mDNS service registration and discovery for Reachy Mini robots."""

from __future__ import annotations

import logging
import re
import socket
import subprocess
import sys
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
        logger.debug("Discovered mDNS service: %s", name)
        info = zc.get_service_info(type_, name)
        if info is None:
            logger.debug("Could not resolve service info for: %s", name)
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

    On macOS, uses the native ``dns-sd`` command because mDNSResponder
    holds port 5353 exclusively. On other platforms, uses the zeroconf
    library directly.

    Args:
        timeout: Maximum time to wait for responses, in seconds.

    Returns:
        A list of discovered robots.

    """
    if sys.platform == "darwin":
        robots = _find_robots_dnssd(timeout)
    else:
        robots = _find_robots_zeroconf(timeout)

    return _filter_alive(robots)


def _find_robots_zeroconf(timeout: float) -> List[DiscoveredRobot]:
    """Browse for robots using the zeroconf library."""
    zc = Zeroconf()

    collector = _RobotCollector()
    browser = ServiceBrowser(zc, SERVICE_TYPE, listener=collector)

    try:
        time.sleep(timeout)
    finally:
        browser.cancel()
        zc.close()

    return collector.robots


def _find_robots_dnssd(timeout: float) -> List[DiscoveredRobot]:
    """Browse for robots using macOS dns-sd command."""
    service_type = "_reachy-mini._tcp"
    robots: List[DiscoveredRobot] = []

    # dns-sd -B never exits on its own — run it, wait, then terminate.
    try:
        proc = subprocess.Popen(
            ["dns-sd", "-B", service_type],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(timeout)
        proc.terminate()
        stdout, _ = proc.communicate(timeout=2)
    except (FileNotFoundError, OSError):
        return []

    # Parse instance names from output lines like:
    # 14:43:00.099  Add  2  14  local.  _reachy-mini._tcp.  reachy_mini
    instance_names: List[str] = []
    for line in stdout.splitlines():
        if "Add" in line and service_type in line:
            parts = line.split()
            if len(parts) >= 7:
                instance_names.append(parts[-1])

    # Step 2: resolve each instance
    for name in instance_names:
        try:
            proc = subprocess.Popen(
                ["dns-sd", "-L", name, service_type],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            time.sleep(0.5)
            proc.terminate()
            stdout, _ = proc.communicate(timeout=1)
        except (FileNotFoundError, OSError):
            continue

        # Parse output like:
        # reachy_mini._reachy-mini._tcp.local. can be reached at reachy-mini.local.:8000
        #  version=1.3.1 robot_name=reachy_mini ws_path=/ws/sdk
        host = ""
        port = 0
        properties: Dict[str, str] = {}

        for line in stdout.splitlines():
            reach_match = re.search(r"can be reached at (.+):(\d+)", line)
            if reach_match:
                host = reach_match.group(1)
                port = int(reach_match.group(2))
            # TXT record line starts with whitespace
            if line.startswith(" ") or line.startswith("\t"):
                for pair in line.split():
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        properties[k] = v

        if host and port:
            # Resolve hostname to IP addresses
            addresses: List[str] = []
            try:
                for addrinfo in socket.getaddrinfo(host, port, socket.AF_INET):
                    addr = str(addrinfo[4][0])
                    if addr not in addresses:
                        addresses.append(addr)
            except socket.gaierror:
                pass

            robot_name = properties.get("robot_name", name)
            robots.append(
                DiscoveredRobot(
                    name=robot_name,
                    host=host,
                    port=port,
                    addresses=addresses,
                    properties=properties,
                )
            )

    return robots


def _filter_alive(robots: List[DiscoveredRobot]) -> List[DiscoveredRobot]:
    """Filter out stale entries by checking TCP connectivity."""
    alive: List[DiscoveredRobot] = []
    for robot in robots:
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
