"""Tests for systemd notification helpers."""

from __future__ import annotations

import os
import socket
from pathlib import Path

from reachy_mini.daemon.systemd import SystemdNotifier


def test_notifier_is_disabled_without_notify_socket() -> None:
    """Non-systemd launches should be a no-op."""
    notifier = SystemdNotifier.from_environment({})

    assert not notifier.enabled
    assert notifier.watchdog_interval_seconds is None
    notifier.ready()


def test_notifier_sends_ready_status_and_watchdog(tmp_path: Path) -> None:
    """Systemd payloads are sent to the configured Unix datagram socket."""
    socket_path = tmp_path / "notify.sock"
    receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    receiver.bind(str(socket_path))
    receiver.settimeout(1.0)

    try:
        notifier = SystemdNotifier(str(socket_path), watchdog_usec=4_000_000)

        notifier.status("Starting test daemon")
        notifier.ready("Test daemon ready")
        notifier.watchdog()

        payloads = [receiver.recv(1024).decode() for _ in range(3)]
    finally:
        receiver.close()

    assert payloads == [
        "STATUS=Starting test daemon",
        "READY=1\nSTATUS=Test daemon ready",
        "WATCHDOG=1",
    ]
    assert notifier.watchdog_interval_seconds == 2.0


def test_watchdog_is_ignored_for_a_different_pid() -> None:
    """WATCHDOG_PID gates heartbeat support the same way systemd does."""
    notifier = SystemdNotifier.from_environment(
        {
            "NOTIFY_SOCKET": "/tmp/systemd-notify.sock",
            "WATCHDOG_USEC": "1000000",
            "WATCHDOG_PID": str(os.getpid() + 1),
        }
    )

    assert notifier.enabled
    assert notifier.watchdog_interval_seconds is None
