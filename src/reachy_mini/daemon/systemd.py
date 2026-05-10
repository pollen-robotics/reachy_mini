"""Small sd_notify helper for systemd-managed daemon launches."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from collections.abc import Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SystemdNotifier:
    """Best-effort client for systemd's notification socket."""

    notify_socket: str | None
    watchdog_usec: int | None = None

    @classmethod
    def from_environment(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "SystemdNotifier":
        """Create a notifier from systemd environment variables."""
        values = env if env is not None else os.environ
        watchdog_usec = cls._parse_watchdog_usec(values)
        return cls(values.get("NOTIFY_SOCKET"), watchdog_usec)

    @staticmethod
    def _parse_watchdog_usec(values: Mapping[str, str]) -> int | None:
        watchdog_pid = values.get("WATCHDOG_PID")
        if watchdog_pid and watchdog_pid != str(os.getpid()):
            return None

        raw_usec = values.get("WATCHDOG_USEC")
        if raw_usec is None:
            return None

        try:
            watchdog_usec = int(raw_usec)
        except ValueError:
            logger.warning("Ignoring invalid WATCHDOG_USEC=%r", raw_usec)
            return None

        return watchdog_usec if watchdog_usec > 0 else None

    @property
    def enabled(self) -> bool:
        """Whether sd_notify messages can be sent."""
        return bool(self.notify_socket)

    @property
    def watchdog_interval_seconds(self) -> float | None:
        """Return a conservative heartbeat interval, if watchdog is configured."""
        if self.watchdog_usec is None:
            return None
        return max(self.watchdog_usec / 2_000_000, 1.0)

    def status(self, message: str) -> None:
        """Publish a human-readable service status message."""
        self.notify(f"STATUS={message}")

    def ready(self, message: str = "Reachy Mini daemon ready") -> None:
        """Tell systemd the service has reached its intended ready point."""
        self.notify(f"READY=1\nSTATUS={message}")

    def stopping(self, message: str = "Stopping Reachy Mini daemon") -> None:
        """Tell systemd the service is shutting down."""
        self.notify(f"STOPPING=1\nSTATUS={message}")

    def watchdog(self) -> None:
        """Send one watchdog heartbeat."""
        self.notify("WATCHDOG=1")

    async def watchdog_loop(self) -> None:
        """Send watchdog heartbeats until cancelled."""
        interval = self.watchdog_interval_seconds
        if interval is None or not self.enabled:
            return

        try:
            while True:
                self.watchdog()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise

    def notify(self, payload: str) -> None:
        """Send a raw sd_notify payload, ignoring missing systemd support."""
        if not self.notify_socket:
            return

        notify_socket = self.notify_socket
        address: str | bytes
        if notify_socket.startswith("@"):
            address = b"\0" + notify_socket[1:].encode()
        else:
            address = notify_socket

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
                sock.sendto(payload.encode(), address)
        except OSError as exc:
            logger.debug("Failed to notify systemd: %s", exc)
