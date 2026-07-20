"""Tests for reachy_mini.daemon.utils.

Pure helpers plus branch coverage for the serial/camera/daemon utilities,
with all serial, psutil, subprocess, socket and filesystem access mocked.
"""

from __future__ import annotations

import pytest

from reachy_mini.daemon import utils


class _FakePort:
    """Minimal stand-in for a serial.tools.list_ports port object."""

    def __init__(self, device: str, hwid: str) -> None:
        self.device = device
        self.hwid = hwid


class _FakeProc:
    """Minimal stand-in for a psutil.Process yielded by process_iter."""

    def __init__(self, pid: int, cmdline: list[str] | None) -> None:
        self.pid = pid
        self.info = {"pid": pid, "name": "python", "cmdline": cmdline}


@pytest.mark.parametrize(
    "ip",
    ["127.0.0.1", "127.0.0.53", "::1", "localhost", "0.0.0.0"],
)
def test_is_localhost_true(ip: str) -> None:
    """Recognized localhost addresses return True."""
    assert utils.is_localhost(ip) is True


@pytest.mark.parametrize(
    "ip",
    ["192.168.1.10", "example.com", "", "10.0.0.1", "128.0.0.1"],
)
def test_is_localhost_false(ip: str) -> None:
    """Non-localhost addresses return False."""
    assert utils.is_localhost(ip) is False


def test_is_localhost_none() -> None:
    """None returns False."""
    assert utils.is_localhost(None) is False


def test_find_serial_port_wireless_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wireless version returns the UART path when it exists."""
    monkeypatch.setattr(utils.os.path, "exists", lambda p: p == "/dev/ttyAMA3")
    assert utils.find_serial_port(wireless_version=True) == ["/dev/ttyAMA3"]


def test_find_serial_port_wireless_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wireless version returns empty list when the UART is absent."""
    monkeypatch.setattr(utils.os.path, "exists", lambda p: False)
    assert utils.find_serial_port(wireless_version=True) == []


def test_find_serial_port_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lite version returns matching device by VID:PID in hwid."""
    ports = [
        _FakePort("/dev/ttyUSB0", "USB VID:PID=1A86:55D3 SER=123"),
        _FakePort("/dev/ttyUSB1", "USB VID:PID=DEAD:BEEF"),
    ]
    monkeypatch.setattr(utils.serial.tools.list_ports, "comports", lambda: ports)
    assert utils.find_serial_port() == ["/dev/ttyUSB0"]


def test_find_serial_port_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lite version returns empty list when no port matches."""
    ports = [_FakePort("/dev/ttyUSB1", "USB VID:PID=DEAD:BEEF")]
    monkeypatch.setattr(utils.serial.tools.list_ports, "comports", lambda: ports)
    assert utils.find_serial_port() == []


def test_is_local_camera_available_linux_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux socket present means camera available."""
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(
        utils.os.path, "exists", lambda p: p == utils.CAMERA_SOCKET_PATH
    )
    assert utils.is_local_camera_available() is True


def test_is_local_camera_available_linux_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux socket absent means camera unavailable."""
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(utils.os.path, "exists", lambda p: False)
    assert utils.is_local_camera_available() is False


def test_daemon_check_no_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing happens when spawn_daemon is False."""
    called = {"popen": False}
    monkeypatch.setattr(
        utils.psutil, "process_iter", lambda *a, **k: iter([])
    )
    monkeypatch.setattr(
        utils.subprocess,
        "Popen",
        lambda *a, **k: called.__setitem__("popen", True),
    )
    utils.daemon_check(spawn_daemon=False, use_sim=False)
    assert called["popen"] is False


def test_daemon_check_already_running_same_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matching running daemon means no new spawn."""
    procs = [_FakeProc(42, ["reachy-mini-daemon"])]
    called = {"popen": False}
    monkeypatch.setattr(utils.psutil, "process_iter", lambda *a, **k: iter(procs))
    monkeypatch.setattr(
        utils.subprocess,
        "Popen",
        lambda *a, **k: called.__setitem__("popen", True),
    )
    utils.daemon_check(spawn_daemon=True, use_sim=False)
    assert called["popen"] is False


def test_daemon_check_different_config_kills_and_respawns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running daemon with a different config is killed then respawned."""
    procs = [_FakeProc(42, ["reachy-mini-daemon", "--sim"])]
    killed = {}
    spawned = {}
    monkeypatch.setattr(utils.psutil, "process_iter", lambda *a, **k: iter(procs))
    monkeypatch.setattr(
        utils.os, "kill", lambda pid, sig: killed.update(pid=pid, sig=sig)
    )
    monkeypatch.setattr(utils.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        utils.subprocess, "Popen", lambda cmd, **k: spawned.update(cmd=cmd)
    )
    utils.daemon_check(spawn_daemon=True, use_sim=False)
    assert killed == {"pid": 42, "sig": 9}
    assert spawned["cmd"] == ["reachy-mini-daemon"]


def test_daemon_check_not_running_spawns_sim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No daemon running spawns a new one with the --sim flag."""
    procs = [_FakeProc(1, ["something-else"]), _FakeProc(2, None)]
    spawned = {}
    monkeypatch.setattr(utils.psutil, "process_iter", lambda *a, **k: iter(procs))
    monkeypatch.setattr(utils.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        utils.subprocess, "Popen", lambda cmd, **k: spawned.update(cmd=cmd)
    )
    utils.daemon_check(spawn_daemon=True, use_sim=True)
    assert spawned["cmd"] == ["reachy-mini-daemon", "--sim"]


def test_daemon_check_skips_inaccessible_proc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Processes raising psutil errors are skipped, still leading to a spawn."""

    class _RaisingProc:
        pid = 99

        @property
        def info(self) -> dict:
            raise utils.psutil.AccessDenied(self.pid)

    procs = [_RaisingProc()]
    spawned = {}
    monkeypatch.setattr(utils.psutil, "process_iter", lambda *a, **k: iter(procs))
    monkeypatch.setattr(
        utils.subprocess, "Popen", lambda cmd, **k: spawned.update(cmd=cmd)
    )
    utils.daemon_check(spawn_daemon=True, use_sim=False)
    assert spawned["cmd"] == ["reachy-mini-daemon"]


def test_get_ip_address_linux_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    """Linux ioctl failure returns None."""
    import socket

    monkeypatch.setattr("platform.system", lambda: "Linux")

    class _FakeSock:
        def fileno(self) -> int:
            return 0

    monkeypatch.setattr(socket, "socket", lambda *a, **k: _FakeSock())
    import fcntl

    def _raise(*a: object, **k: object) -> None:
        raise OSError("no such interface")

    monkeypatch.setattr(fcntl, "ioctl", _raise)
    assert utils.get_ip_address("wlan0") is None


def test_get_ip_address_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported platform returns None."""
    monkeypatch.setattr("platform.system", lambda: "SunOS")
    assert utils.get_ip_address() is None
