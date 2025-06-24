import os
import subprocess
import sys
import time
from typing import Callable, Optional

import numpy as np
import psutil
import serial.tools.list_ports
from scipy.spatial.transform import Rotation as R


def create_head_pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0, mm=False, degrees=True):
    pose = np.eye(4)
    rot = R.from_euler("xyz", [roll, pitch, yaw], degrees=degrees).as_matrix()
    pose[:3, :3] = rot
    pose[:, 3] = [x, y, z, 0]
    if mm:
        pose[:3, 3] /= 1000

    return pose


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


def linear_pose_interpolation(
    start_pose: np.ndarray, target_pose: np.ndarray, t: float
):
    # Extract rotations
    rot_start = R.from_matrix(start_pose[:3, :3])
    rot_end = R.from_matrix(target_pose[:3, :3])

    # Compute relative rotation q_rel such that rot_start * q_rel = rot_end
    q_rel = rot_start.inv() * rot_end
    # Convert to rotation vector (axis-angle)
    rotvec_rel = q_rel.as_rotvec()
    # Scale the rotation vector by t (allows t<0 or >1 for overshoot)
    rot_interp = (rot_start * R.from_rotvec(rotvec_rel * t)).as_matrix()

    # Extract translations
    pos_start = start_pose[:3, 3]
    pos_end = target_pose[:3, 3]
    # Linear interpolation/extrapolation on translation
    pos_interp = pos_start + (pos_end - pos_start) * t

    # Compose homogeneous transformation
    interp_pose = np.eye(4)
    interp_pose[:3, :3] = rot_interp
    interp_pose[:3, 3] = pos_interp

    return interp_pose


def time_trajectory(t: float, method="default"):
    method = "minjerk" if method == "default" else method

    if t < 0 or t > 1:
        raise ValueError("time value is out of range [0,1]")

    match method:
        case "linear":
            return t

        case "minjerk":
            return 10 * t**3 - 15 * t**4 + 6 * t**5

        case "ease":
            if t < 0.5:
                return 2 * t * t
            else:
                return 1 - ((-2 * t + 2) ** 2) / 2

        case "cartoon":
            c1 = 1.70158
            c2 = c1 * 1.525

            if t < 0.5:
                # phase in
                return ((2 * t) ** 2 * ((c2 + 1) * 2 * t - c2)) / 2
            else:
                # phase out
                return (((2 * t - 2) ** 2 * ((c2 + 1) * (2 * t - 2) + c2)) + 2) / 2

        case _:
            raise ValueError(
                f"Unknown interpolation method: {method} (possible values: linear, minjerk, ease, cartoon)"
            )


def create_pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0, mm=False, degrees=True):
    pose = np.eye(4)
    rot = R.from_euler("xyz", [roll, pitch, yaw], degrees=degrees).as_matrix()
    pose[:3, :3] = rot
    pose[:, 3] = [x, y, z, 0]
    if mm:
        pose[:3, 3] /= 1000

    pose[2, 3] += 0.177  # :(
    return pose


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
