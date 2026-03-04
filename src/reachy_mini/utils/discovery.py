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


def _get_local_ip() -> str:
    """Get the primary local IP address (works without internet)."""
    try:
        # Use a link-local multicast address — no actual packet is sent,
        # but the OS picks the interface that would route to the LAN.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("224.0.0.1", 0))
            ip: str = s.getsockname()[0]
            return ip
        finally:
            s.close()
    except OSError:
        return "127.0.0.1"


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
            "address": _get_local_ip(),
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
            self._zeroconf.register_service(self._info, allow_name_change=True)
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


def find_robots(timeout: float = 5.0) -> List[DiscoveredRobot]:
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

    instance_names = _dnssd_browse(service_type, timeout)

    for name in instance_names:
        resolved = _dnssd_resolve(name, service_type)
        if resolved is not None:
            robots.append(resolved)

    return robots


def _dnssd_browse(service_type: str, timeout: float) -> List[str]:
    """Run dns-sd -B and collect instance names as they arrive."""
    try:
        proc = subprocess.Popen(
            ["dns-sd", "-B", service_type],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )
    except (FileNotFoundError, OSError):
        return []

    assert proc.stdout is not None
    lines: List[str] = []

    def reader() -> None:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            lines.append(raw_line.decode(errors="replace"))

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    time.sleep(timeout)
    proc.terminate()
    t.join(timeout=1)

    instance_names: List[str] = []
    for line in lines:
        if "Add" in line and service_type in line:
            # Instance name is everything after the service type column
            # and may contain spaces (e.g. "reachy_mini (2)")
            idx = line.find(service_type)
            if idx >= 0:
                name = line[idx + len(service_type) :].lstrip(".").strip()
                if name and name not in instance_names:
                    instance_names.append(name)

    return instance_names


def _dnssd_resolve(name: str, service_type: str) -> DiscoveredRobot | None:
    """Run dns-sd -L to resolve a single service instance."""
    try:
        proc = subprocess.Popen(
            ["dns-sd", "-L", name, service_type],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )
    except (FileNotFoundError, OSError):
        return None

    assert proc.stdout is not None
    lines: List[str] = []

    def reader() -> None:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            lines.append(raw_line.decode(errors="replace"))

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    time.sleep(0.5)
    proc.terminate()
    t.join(timeout=1)

    # Parse output like:
    # reachy_mini._reachy-mini._tcp.local. can be reached at reachy-mini.local.:8000
    #  version=1.3.1 robot_name=reachy_mini ws_path=/ws/sdk
    host = ""
    port = 0
    properties: Dict[str, str] = {}

    for line in lines:
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

    if not host or not port:
        return None

    # Prefer the explicit address from the TXT record (handles same-hostname Pis)
    addresses: List[str] = []
    if "address" in properties:
        addresses.append(properties["address"])
    else:
        try:
            for addrinfo in socket.getaddrinfo(host, port, socket.AF_INET):
                addr = str(addrinfo[4][0])
                if addr not in addresses:
                    addresses.append(addr)
        except socket.gaierror:
            pass

    robot_name = properties.get("robot_name", name)
    return DiscoveredRobot(
        name=robot_name,
        host=host,
        port=port,
        addresses=addresses,
        properties=properties,
    )


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
    parser.add_argument("--timeout", type=float, default=5.0, help="Discovery timeout in seconds (default: 5.0)")
    args = parser.parse_args()

    robots = find_robots(timeout=args.timeout)

    if not robots:
        print("No robots found.")
    else:
        for robot in robots:
            addrs = ", ".join(robot.addresses)
            print(f"{robot.name} - {addrs}:{robot.port} ({robot.host})")
