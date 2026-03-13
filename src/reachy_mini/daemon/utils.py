"""Utilities for managing the Reachy Mini daemon."""

import os
import struct
import subprocess
import time

import psutil
import serial.tools.list_ports

# Path to the unix socket created by WebRTC daemon for local camera access (Linux/macOS)
CAMERA_SOCKET_PATH = "/tmp/reachymini_camera_socket"

# Named pipe for local camera access on Windows.
# GStreamer's win32ipcvideosink / win32ipcvideosrc expect the full
# Windows named-pipe path (e.g. \\.\pipe\<name>).
CAMERA_PIPE_NAME = "\\\\.\\pipe\\reachymini_camera_pipe"


def is_localhost(ip: str | None) -> bool:
    """Check if an IP address corresponds to localhost.

    Args:
        ip: The IP address to check. Can be None.

    Returns:
        True if the IP is a localhost address, False otherwise.

    """
    if ip is None:
        return False

    localhost_addresses = {
        "127.0.0.1",
        "::1",
        "localhost",
        "0.0.0.0",
    }
    return ip in localhost_addresses or ip.startswith("127.")


def is_local_camera_available() -> bool:
    """Check if local camera access is available via IPC.

    The WebRTC daemon exposes raw camera frames via a local IPC mechanism
    so that on-device clients can read frames without WebRTC overhead:
    - Linux/macOS: Unix domain socket at CAMERA_SOCKET_PATH
    - Windows: Named pipe at CAMERA_PIPE_NAME

    Returns:
        True if the local camera IPC endpoint exists and is accessible.

    """
    import platform

    if platform.system() == "Windows":
        return _win32_pipe_exists(CAMERA_PIPE_NAME)
    else:
        return os.path.exists(CAMERA_SOCKET_PATH)


def _win32_pipe_exists(pipe_path: str) -> bool:
    """Check if a Windows named pipe exists by attempting to open it.

    ``os.path.exists()`` is unreliable for Windows named pipes — it may
    return ``False`` even when the pipe is live and connectable.  Using
    the Win32 ``CreateFileW`` API directly gives a definitive answer.

    Returns:
        True if the pipe can be opened (exists), False otherwise.

    """
    import ctypes

    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    GENERIC_READ = 0x80000000
    OPEN_EXISTING = 3

    handle = ctypes.windll.kernel32.CreateFileW(  # type: ignore[attr-defined]
        pipe_path, GENERIC_READ, 0, None, OPEN_EXISTING, 0, None
    )
    if handle == INVALID_HANDLE_VALUE:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
    return True


def daemon_check(spawn_daemon: bool, use_sim: bool) -> None:
    """Check if the Reachy Mini daemon is running and spawn it if necessary."""

    def is_python_script_running(
        script_name: str,
    ) -> tuple[bool, int | None, bool | None]:
        """Check if a specific Python script is running."""
        found_script = False
        simluation_enabled = False
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                safe_cmdline = proc.info.get("cmdline") or []
                for cmd in safe_cmdline:
                    if script_name in cmd:
                        found_script = True
                    if "--sim" in cmd:
                        simluation_enabled = True
                if found_script:
                    return True, proc.pid, simluation_enabled
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return False, None, None

    if spawn_daemon:
        daemon_is_running, pid, sim = is_python_script_running("reachy-mini-daemon")
        if daemon_is_running and sim == use_sim:
            print(
                f"Reachy Mini daemon is already running (PID: {pid}). "
                "No need to spawn a new one."
            )
            return
        elif daemon_is_running and sim != use_sim:
            print(
                f"Reachy Mini daemon is already running (PID: {pid}) with a different configuration. "
            )
            print("Killing the existing daemon...")
            assert pid is not None, "PID should not be None if daemon is running"
            os.kill(pid, 9)
            time.sleep(1)

        print("Starting a new daemon...")
        subprocess.Popen(
            ["reachy-mini-daemon", "--sim"] if use_sim else ["reachy-mini-daemon"],
            start_new_session=True,
        )


def find_serial_port(
    wireless_version: bool = False,
    vid: str = "1a86",
    pid: str = "55d3",
    pi_uart: str = "/dev/ttyAMA3",
) -> list[str]:
    """Find the serial port for Reachy Mini based on VID and PID or the Raspberry Pi UART for the wireless version.

    Args:
        wireless_version (bool): Whether to look for the wireless version using the Raspberry Pi UART.
        vid (str): Vendor ID of the device. (eg. "1a86").
        pid (str): Product ID of the device. (eg. "55d3").
        pi_uart (str): Path to the Raspberry Pi UART device. (eg. "/dev/ttyAMA3").

    """
    # If it's a wireless version, we should use the Raspberry Pi UART
    if wireless_version:
        return [pi_uart] if os.path.exists(pi_uart) else []

    # If it's a lite version, we should find it using the VID and PID
    ports = serial.tools.list_ports.comports()

    vid = vid.upper()
    pid = pid.upper()

    return [p.device for p in ports if f"USB VID:PID={vid}:{pid}" in p.hwid]


def get_ip_address(ifname: str = "wlan0") -> str | None:
    """Get the IP address of a specific network interface (Linux and Windows)."""
    import platform
    import socket

    if platform.system() == "Linux":
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            import fcntl

            return socket.inet_ntoa(
                fcntl.ioctl(
                    s.fileno(),
                    0x8915,  # SIOCGIFADDR
                    struct.pack("256s", ifname[:15].encode("utf-8")),
                )[20:24]
            )
        except OSError:
            print(f"Could not get IP address for interface {ifname}.")
            return None
    elif platform.system() == "Windows":
        import psutil

        addrs = psutil.net_if_addrs()
        if ifname in addrs:
            for snic in addrs[ifname]:
                if snic.family == socket.AF_INET:
                    return str(snic.address)
        print(f"Could not get IP address for interface {ifname} on Windows.")
        return None
    else:
        print(f"Platform {platform.system()} not supported for get_ip_address.")
        return None
