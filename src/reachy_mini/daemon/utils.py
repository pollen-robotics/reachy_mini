"""Utilities for managing the Reachy Mini daemon."""

import fcntl
import os
import socket
import struct
import subprocess
import time

import serial.tools.list_ports

import psutil


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
                for cmd in proc.info["cmdline"]:
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

def find_serial_port(vid: str = "1a86", pid: str = "55d3") -> list[str]:
    """Find the serial port for Reachy Mini based on VID and PID.

    Args:
        vid (str): Vendor ID of the device. (eg. "1a86").
        pid (str): Product ID of the device. (eg. "55d3").

    """
    ports = serial.tools.list_ports.comports()

    vid = vid.upper()
    pid = pid.upper()

    return [p.device for p in ports if f"USB VID:PID={vid}:{pid}" in p.hwid]

def get_ip_address(ifname: str = "wlan0") -> str | None:
    """Get the IP address of a specific network interface (Linux Only)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
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

