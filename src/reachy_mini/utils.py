import os
import subprocess
import sys
import time
from typing import Callable, Optional

import numpy as np
import psutil
import serial.tools.list_ports

InterpolationFunc = Callable[[float], np.ndarray]


def minimum_jerk(
    starting_position: np.ndarray,
    goal_position: np.ndarray,
    duration: float,
    starting_velocity: Optional[np.ndarray] = None,
    starting_acceleration: Optional[np.ndarray] = None,
    final_velocity: Optional[np.ndarray] = None,
    final_acceleration: Optional[np.ndarray] = None,
) -> InterpolationFunc:
    """Compute the mimimum jerk interpolation function from starting position to goal position."""
    if starting_velocity is None:
        starting_velocity = np.zeros(starting_position.shape)
    if starting_acceleration is None:
        starting_acceleration = np.zeros(starting_position.shape)
    if final_velocity is None:
        final_velocity = np.zeros(goal_position.shape)
    if final_acceleration is None:
        final_acceleration = np.zeros(goal_position.shape)

    a0 = starting_position
    a1 = starting_velocity
    a2 = starting_acceleration / 2

    d1, d2, d3, d4, d5 = [duration**i for i in range(1, 6)]

    A = np.array(((d3, d4, d5), (3 * d2, 4 * d3, 5 * d4), (6 * d1, 12 * d2, 20 * d3)))
    B = np.array(
        (
            goal_position - a0 - (a1 * d1) - (a2 * d2),
            final_velocity - a1 - (2 * a2 * d1),
            final_acceleration - (2 * a2),
        )
    )
    X = np.linalg.solve(A, B)

    coeffs = [a0, a1, a2, X[0], X[1], X[2]]

    def f(t: float) -> np.ndarray:
        if t > duration:
            return goal_position
        return np.sum([c * t**i for i, c in enumerate(coeffs)], axis=0)

    return f


def daemon_check(spawn_daemon, use_sim):
    def is_python_script_running(script_name):
        """Check if a specific Python script is running"""
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
            os.kill(pid, 9)
            time.sleep(1)

        print("Starting a new daemon...")
        subprocess.Popen(
            ["reachy-mini-daemon", "--sim"] if use_sim else ["reachy-mini-daemon"],
            start_new_session=True,
        )


def find_arduino_nano_ch340g() -> Optional[str]:
    """
    Simple, efficient detector for Arduino Nano with CH340G.
    Returns port path or None.
    """
    # CH340G identifiers
    CH340_VID = 0x1A86
    CH340_PIDS = [0x7523, 0x5523]

    try:
        for port in serial.tools.list_ports.comports():
            # Direct comparison - most efficient
            if port.vid == CH340_VID and port.pid in CH340_PIDS:
                return port.device
    except:
        pass

    return None
