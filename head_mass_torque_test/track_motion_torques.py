#!/usr/bin/env python3
"""
Track Motor Torques During Motion Between Poses

Unlike the static pose tests, this script tracks torques continuously
during the motion/transition between poses to understand dynamic loads.

Usage:
    python track_motion_torques.py
"""

import csv
import select
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import serial.tools.list_ports
from rustypot import Xl330PyController

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reachy_mini.kinematics.analytical_kinematics import AnalyticalKinematics
from reachy_mini.utils import create_head_pose

# ============================================================================
# CONFIGURATION
# ============================================================================

# Motor configuration
MOTOR_IDS = {
    "body_rotation": 10,
    "stewart_1": 11,
    "stewart_2": 12,
    "stewart_3": 13,
    "stewart_4": 14,
    "stewart_5": 15,
    "stewart_6": 16,
}

# Test pose sequence - robot will move through these poses in order
TEST_POSE_SEQUENCE = [
    {
        "key": "neutral",
        "name": "Neutral (0°, 0°)",
        "pose": create_head_pose(roll=0, pitch=0, yaw=0, degrees=True),
    },
    {
        "key": "pitch_up_20",
        "name": "Pitch up 20°",
        "pose": create_head_pose(roll=0, pitch=-20, yaw=0, degrees=True),
    },
    {
        "key": "neutral_2",
        "name": "Neutral (return)",
        "pose": create_head_pose(roll=0, pitch=0, yaw=0, degrees=True),
    },
    {
        "key": "pitch_down_20",
        "name": "Pitch down 20°",
        "pose": create_head_pose(roll=0, pitch=20, yaw=0, degrees=True),
    },
    {
        "key": "neutral_3",
        "name": "Neutral (return)",
        "pose": create_head_pose(roll=0, pitch=0, yaw=0, degrees=True),
    },
    {
        "key": "roll_right_20",
        "name": "Roll right 20°",
        "pose": create_head_pose(roll=20, pitch=0, yaw=0, degrees=True),
    },
    {
        "key": "neutral_4",
        "name": "Neutral (return)",
        "pose": create_head_pose(roll=0, pitch=0, yaw=0, degrees=True),
    },
    {
        "key": "roll_left_20",
        "name": "Roll left 20°",
        "pose": create_head_pose(roll=-20, pitch=0, yaw=0, degrees=True),
    },
    {
        "key": "neutral_5",
        "name": "Neutral (final)",
        "pose": create_head_pose(roll=0, pitch=0, yaw=0, degrees=True),
    },
]

# Torque constants (motor spec: 0.345 Nm/A)
K_NM_TO_MA = 1 / 0.345 * 1000  # = 2898.55 mA/Nm
CORRECTION_FACTOR = 1.0
EFFICIENCY = 1.0

# Safety limits
STALL_TORQUE_NM = 0.6
SAFE_LIMIT_NM = 0.45

# Motion parameters
MOTION_DURATION = 0.25  # seconds per transition
MOTION_FREQUENCY = 50  # Hz - control loop frequency

# Monitoring
MONITORING_FREQUENCY = 100  # Hz - higher frequency for motion tracking

# PID Configuration
# Default PID values for stewart motors
DEFAULT_PID = {
    "p_gain": 1000,
    "i_gain": 100,
    "d_gain": 0,  # D gain is typically 0 for position control
}

# PID values per payload (in grams)
# If a payload is not in this dict, DEFAULT_PID will be used
PAYLOAD_PID_CONFIG = {
    # Example: different PID for different payloads
    0: {"p_gain": 1000, "i_gain": 100, "d_gain": 0},
    # 100: {"p_gain": 1200, "i_gain": 120, "d_gain": 0},
    # 200: {"p_gain": 1400, "i_gain": 140, "d_gain": 0},
    250: {"p_gain": 1500, "i_gain": 150, "d_gain": 0},
    300: {"p_gain": 2000, "i_gain": 200, "d_gain": 0},
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def current_to_torque(current_ma):
    """Convert current (mA) to torque (N·m)."""
    return current_ma * EFFICIENCY * CORRECTION_FACTOR / K_NM_TO_MA


def setup_motors(port="/dev/ttyACM0"):
    """Initialize motors with rustypot."""
    print(f"\nConnecting to motors on {port}...")

    try:
        print("  Creating controller (timeout: 1.0s, baudrate: 1000000)...")
        controller = Xl330PyController(port, 1000000, 1.0)
        time.sleep(0.5)
        print("  ✓ Controller created")

        # Enable torque on all motors
        motor_ids = list(MOTOR_IDS.values())
        print(f"  Enabling {len(motor_ids)} motors...")

        for motor_id in motor_ids:
            try:
                controller.write_torque_enable(motor_id, True)
                print(f"    ✓ Motor ID {motor_id}")
            except Exception as e:
                print(f"    ✗ Motor ID {motor_id}: {e}")

        print(f"  ✓ Setup complete")
        return controller
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


def set_motor_pid(controller, motor_id, p_gain, i_gain, d_gain):
    """Set PID gains for a motor."""
    try:
        controller.write_position_p_gain(motor_id, p_gain)
        controller.write_position_i_gain(motor_id, i_gain)
        controller.write_position_d_gain(motor_id, d_gain)
        return True
    except Exception as e:
        print(f"    ✗ Failed to set PID for motor {motor_id}: {e}")
        return False


def configure_pid_for_payload(controller, payload_g):
    """Configure PID values for all stewart motors based on payload."""
    # Get PID config for this payload, or use default
    pid_config = PAYLOAD_PID_CONFIG.get(payload_g, DEFAULT_PID)

    p_gain = pid_config["p_gain"]
    i_gain = pid_config["i_gain"]
    d_gain = pid_config["d_gain"]

    print(f"\nConfiguring PID for {payload_g}g payload:")
    print(f"  P={p_gain}, I={i_gain}, D={d_gain}")

    stewart_ids = [MOTOR_IDS[f"stewart_{i}"] for i in range(1, 7)]
    success_count = 0

    for motor_id in stewart_ids:
        if set_motor_pid(controller, motor_id, p_gain, i_gain, d_gain):
            success_count += 1
            print(f"    ✓ Motor {motor_id}")

    if success_count == len(stewart_ids):
        print(f"  ✓ PID configured for all motors")
    else:
        print(f"  ⚠ PID configured for {success_count}/{len(stewart_ids)} motors")

    return success_count == len(stewart_ids)


class MotionMonitor:
    """Continuous motor monitoring during motion."""

    def __init__(self, controller, frequency=100):
        self.controller = controller
        self.frequency = frequency
        self.running = False
        self.thread = None
        self.data = []
        self.start_time = None
        self.stewart_ids = [MOTOR_IDS[f"stewart_{i}"] for i in range(1, 7)]
        self.body_rotation_id = MOTOR_IDS["body_rotation"]
        self.lock = threading.Lock()
        self.target_positions = None  # Current target positions for error calculation

    def start(self):
        """Start monitoring."""
        self.running = True
        self.data = []
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def set_target_positions(self, target_positions):
        """Update the target positions for error calculation."""
        with self.lock:
            self.target_positions = np.array(target_positions, dtype=float)

    def _monitor_loop(self):
        """Background monitoring loop."""
        period = 1.0 / self.frequency

        while self.running:
            loop_start = time.time()

            try:
                # Read current and position from all stewart motors
                currents = []
                positions = []
                temperatures = []

                for motor_id in self.stewart_ids:
                    current = self.controller.read_present_current(motor_id)
                    pos = self.controller.read_present_position(motor_id)
                    temp = self.controller.read_present_temperature(motor_id)

                    # Extract scalar from array if needed
                    if isinstance(current, (list, np.ndarray)):
                        current = float(current[0])
                    if isinstance(pos, (list, np.ndarray)):
                        pos = float(pos[0])
                    if isinstance(temp, (list, np.ndarray)):
                        temp = float(temp[0])

                    currents.append(current)
                    positions.append(pos)
                    temperatures.append(temp)

                # Also read body rotation
                body_pos = self.controller.read_present_position(self.body_rotation_id)
                if isinstance(body_pos, (list, np.ndarray)):
                    body_pos = float(body_pos[0])

                elapsed = time.time() - self.start_time
                torques = [current_to_torque(c) for c in currents]
                max_abs_current = max(abs(c) for c in currents)
                max_abs_torque = max(abs(t) for t in torques)
                max_temp = max(temperatures)

                # Calculate position errors if target is set
                with self.lock:
                    position_errors = None
                    target_pos = None
                    max_pos_error = None
                    rms_pos_error = None

                    if self.target_positions is not None:
                        # Error for stewart motors only (skip body rotation)
                        target_pos = self.target_positions[
                            1:
                        ].tolist()  # Stewart motors
                        position_errors = [
                            target_pos[i] - positions[i] for i in range(6)
                        ]
                        max_pos_error = max(abs(e) for e in position_errors)
                        rms_pos_error = float(
                            np.sqrt(np.mean(np.array(position_errors) ** 2))
                        )

                    self.data.append(
                        {
                            "time": elapsed,
                            "currents": currents,
                            "positions": positions,
                            "body_position": body_pos,
                            "target_positions": target_pos,
                            "position_errors": position_errors,
                            "temperatures": temperatures,
                            "torques": torques,
                            "max_current": max_abs_current,
                            "max_torque": max_abs_torque,
                            "max_temp": max_temp,
                            "max_pos_error": max_pos_error,
                            "rms_pos_error": rms_pos_error,
                        }
                    )
            except Exception as e:
                print(f"\nWarning: Failed to read motor data: {e}")

            # Sleep for remaining time
            elapsed = time.time() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_latest(self):
        """Get latest reading."""
        with self.lock:
            if self.data:
                return self.data[-1]
        return None

    def get_all_data(self):
        """Get all collected data."""
        with self.lock:
            return self.data.copy()


def move_with_monitoring(
    controller, monitor, current_pos, target_pos, duration=3.0, transition_name=""
):
    """
    Move motors to target positions smoothly while monitoring torques.

    This is the key function that tracks torques DURING motion.
    """
    print(f"\n  Starting transition: {transition_name}")
    print(f"    Duration: {duration}s")
    print(f"    Monitoring frequency: {monitor.frequency} Hz")

    current_pos = np.array(current_pos, dtype=float)
    target_pos = np.array(target_pos, dtype=float)

    motor_ids = [MOTOR_IDS["body_rotation"]] + [
        MOTOR_IDS[f"stewart_{j}"] for j in range(1, 7)
    ]

    # Start monitoring
    monitor.start()

    # Interpolate and move
    steps = int(duration * MOTION_FREQUENCY)
    period = 1.0 / MOTION_FREQUENCY
    print(f"    Total steps: {steps}")

    for i in range(steps):
        loop_start = time.time()

        alpha = i / steps if steps > 0 else 1.0
        pos = current_pos + alpha * (target_pos - current_pos)

        # Update target positions in monitor for error tracking
        monitor.set_target_positions(pos)

        # Set positions
        for j, motor_id in enumerate(motor_ids):
            val = pos[j]
            if isinstance(val, np.ndarray):
                val = val.item()
            controller.write_goal_position(motor_id, float(val))

        if i % 25 == 0:
            latest = monitor.get_latest()
            if latest:
                err_str = (
                    f" | err: {np.rad2deg(latest['max_pos_error']):.3f}°"
                    if latest.get("max_pos_error") is not None
                    else ""
                )
                print(
                    f"      Progress: {i}/{steps} | τ_max: {latest['max_torque']:.3f} N·m | T_max: {latest['max_temp']:.0f}°C{err_str}",
                    end="\r",
                )

        # Sleep for remaining time in this period
        elapsed = time.time() - loop_start
        sleep_time = period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print(f"      Progress: {steps}/{steps} - Done!                              ")

    # Stop monitoring
    monitor.stop()
    motion_data = monitor.get_all_data()

    print(f"    ✓ Collected {len(motion_data)} samples during motion")

    if motion_data:
        max_torque = max(d["max_torque"] for d in motion_data)
        avg_torque = np.mean([d["max_torque"] for d in motion_data])
        print(
            f"    ✓ Torque during motion: max={max_torque:.3f} N·m, avg={avg_torque:.3f} N·m"
        )

        # Position error statistics
        pos_errors = [
            d.get("max_pos_error")
            for d in motion_data
            if d.get("max_pos_error") is not None
        ]
        if pos_errors:
            max_pos_err = max(pos_errors)
            avg_pos_err = np.mean(pos_errors)
            print(
                f"    ✓ Position error: max={np.rad2deg(max_pos_err):.3f}°, "
                f"avg={np.rad2deg(avg_pos_err):.3f}°"
            )

    return motion_data


def get_current_positions(controller):
    """Read current motor positions."""
    motor_ids = [MOTOR_IDS["body_rotation"]] + [
        MOTOR_IDS[f"stewart_{j}"] for j in range(1, 7)
    ]

    positions = []
    for motor_id in motor_ids:
        pos = controller.read_present_position(motor_id)
        if isinstance(pos, (list, np.ndarray)):
            pos = float(pos[0])
        positions.append(pos)

    return np.array(positions, dtype=float)


def save_motion_data_to_csv(all_motion_data, csv_file, payload_g, transition_names):
    """Save all motion data to CSV."""
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(
            [
                "transition",
                "payload_g",
                "time_s",
                "current_m1_ma",
                "current_m2_ma",
                "current_m3_ma",
                "current_m4_ma",
                "current_m5_ma",
                "current_m6_ma",
                "pos_m1_rad",
                "pos_m2_rad",
                "pos_m3_rad",
                "pos_m4_rad",
                "pos_m5_rad",
                "pos_m6_rad",
                "pos_body_rad",
                "target_m1_rad",
                "target_m2_rad",
                "target_m3_rad",
                "target_m4_rad",
                "target_m5_rad",
                "target_m6_rad",
                "error_m1_deg",
                "error_m2_deg",
                "error_m3_deg",
                "error_m4_deg",
                "error_m5_deg",
                "error_m6_deg",
                "temp_m1_c",
                "temp_m2_c",
                "temp_m3_c",
                "temp_m4_c",
                "temp_m5_c",
                "temp_m6_c",
                "torque_m1_nm",
                "torque_m2_nm",
                "torque_m3_nm",
                "torque_m4_nm",
                "torque_m5_nm",
                "torque_m6_nm",
                "max_current_ma",
                "max_torque_nm",
                "max_temp_c",
                "max_pos_error_deg",
                "rms_pos_error_deg",
            ]
        )

        # Data rows for each transition
        for transition_idx, (transition_name, data) in enumerate(
            zip(transition_names, all_motion_data)
        ):
            for sample in data:
                # Handle None values for target positions and errors
                target_positions = sample.get("target_positions") or [None] * 6
                position_errors_rad = sample.get("position_errors") or [None] * 6

                # Convert errors to degrees
                position_errors_deg = [
                    np.rad2deg(err) if err is not None else None
                    for err in position_errors_rad
                ]

                max_pos_error = sample.get("max_pos_error")
                max_pos_error_deg = (
                    np.rad2deg(max_pos_error) if max_pos_error is not None else None
                )

                rms_pos_error = sample.get("rms_pos_error")
                rms_pos_error_deg = (
                    np.rad2deg(rms_pos_error) if rms_pos_error is not None else None
                )

                writer.writerow(
                    [
                        transition_name,
                        payload_g,
                        sample["time"],
                        *sample["currents"],
                        *sample["positions"],
                        sample["body_position"],
                        *target_positions,
                        *position_errors_deg,
                        *sample["temperatures"],
                        *sample["torques"],
                        sample["max_current"],
                        sample["max_torque"],
                        sample["max_temp"],
                        max_pos_error_deg,
                        rms_pos_error_deg,
                    ]
                )


def generate_motion_graphs(
    all_motion_data, output_file, payload_g, transition_names, pose_achievements=None
):
    """
    Generate comprehensive graphs for motion data.

    Args:
        all_motion_data: List of motion data for each transition
        output_file: Path to save the graph
        payload_g: Payload mass in grams
        transition_names: Names of each transition
        pose_achievements: List of (time, pose_name) tuples indicating when poses were achieved
    """
    if not all_motion_data or not any(all_motion_data):
        return

    # Combine all data with transition markers
    fig, axes = plt.subplots(6, 1, figsize=(16, 20))

    # Prepare data arrays
    all_times = []
    all_currents = []
    all_torques = []
    all_positions = []
    all_temps = []
    all_errors = []
    transition_boundaries = []

    current_time_offset = 0

    for transition_idx, (data, name) in enumerate(
        zip(all_motion_data, transition_names)
    ):
        if not data:
            continue

        times = np.array([d["time"] for d in data]) + current_time_offset
        currents = np.array([d["currents"] for d in data])
        torques = np.array([d["torques"] for d in data])
        positions = np.array([d["positions"] for d in data])
        temps = np.array([d["temperatures"] for d in data])

        # Collect position errors (handle None values)
        errors = []
        for d in data:
            if d.get("position_errors") is not None:
                errors.append(d["position_errors"])
            else:
                errors.append([0.0] * 6)  # Default to zero if no error data
        errors = np.array(errors)

        all_times.append(times)
        all_currents.append(currents)
        all_torques.append(torques)
        all_positions.append(positions)
        all_temps.append(temps)
        all_errors.append(errors)

        # Mark transition boundary
        if len(times) > 0:
            transition_boundaries.append((times[-1], name))
            current_time_offset = times[-1]

    # Concatenate all arrays
    all_times = np.concatenate(all_times) if all_times else np.array([])
    all_currents = np.vstack(all_currents) if all_currents else np.array([])
    all_torques = np.vstack(all_torques) if all_torques else np.array([])
    all_positions = np.vstack(all_positions) if all_positions else np.array([])
    all_temps = np.vstack(all_temps) if all_temps else np.array([])
    all_errors = np.vstack(all_errors) if all_errors else np.array([])

    if len(all_times) == 0:
        return

    # Helper function to add pose achievement markers
    def add_pose_markers(ax):
        """Add vertical lines and labels for achieved poses."""
        if pose_achievements:
            for i, (time, pose_name) in enumerate(pose_achievements):
                ax.axvline(
                    time, color="green", linestyle="-.", linewidth=1.5, alpha=0.7
                )
                # Add pose name at the top (rotate for readability)
                y_pos = ax.get_ylim()[1] * 0.95
                ax.text(
                    time,
                    y_pos,
                    f" {pose_name}",
                    rotation=90,
                    verticalalignment="top",
                    fontsize=7,
                    color="green",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="green",
                    ),
                )

    # Plot 1: Motor Torques over entire sequence
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    for i in range(6):
        ax.plot(
            all_times,
            all_torques[:, i],
            label=f"Motor {i+1}",
            alpha=0.8,
            color=colors[i],
        )

    ax.axhline(
        SAFE_LIMIT_NM, color="red", linestyle="--", linewidth=2, label="Safe Limit"
    )
    ax.axhline(-SAFE_LIMIT_NM, color="red", linestyle="--", linewidth=2)

    # Add transition boundaries
    for boundary_time, name in transition_boundaries:
        ax.axvline(boundary_time, color="gray", linestyle=":", alpha=0.5)

    add_pose_markers(ax)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (N·m)")
    ax.set_title(f"Motor Torques During Motion Sequence ({payload_g}g payload)")
    ax.legend(ncol=7, fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 2: Maximum torque envelope
    ax = axes[1]
    max_torques = np.max(np.abs(all_torques), axis=1)
    ax.plot(all_times, max_torques, linewidth=2, color="darkred", label="Max Torque")
    ax.axhline(
        SAFE_LIMIT_NM, color="red", linestyle="--", linewidth=2, label="Safe Limit"
    )

    # Highlight which motor is max at each point
    for i in range(6):
        is_max = np.abs(all_torques[:, i]) == np.max(np.abs(all_torques), axis=1)
        ax.scatter(
            all_times[is_max],
            max_torques[is_max],
            s=10,
            alpha=0.3,
            color=colors[i],
            label=f"M{i+1} peak",
        )

    for boundary_time, name in transition_boundaries:
        ax.axvline(boundary_time, color="gray", linestyle=":", alpha=0.5)

    add_pose_markers(ax)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Max Torque (N·m)")
    ax.set_title("Maximum Torque Envelope")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 3: Motor Positions
    ax = axes[2]
    for i in range(6):
        ax.plot(
            all_times,
            all_positions[:, i],
            label=f"Motor {i+1}",
            alpha=0.7,
            color=colors[i],
        )

    for boundary_time, name in transition_boundaries:
        ax.axvline(boundary_time, color="gray", linestyle=":", alpha=0.5)

    add_pose_markers(ax)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (rad)")
    ax.set_title("Motor Positions During Motion")
    ax.legend(ncol=6, fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 4: Motor Currents
    ax = axes[3]
    for i in range(6):
        ax.plot(
            all_times,
            all_currents[:, i],
            label=f"Motor {i+1}",
            alpha=0.7,
            color=colors[i],
        )

    for boundary_time, name in transition_boundaries:
        ax.axvline(boundary_time, color="gray", linestyle=":", alpha=0.5)

    add_pose_markers(ax)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (mA)")
    ax.set_title("Motor Currents During Motion")
    ax.legend(ncol=6, fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 5: Temperatures
    ax = axes[4]
    for i in range(6):
        ax.plot(
            all_times, all_temps[:, i], label=f"Motor {i+1}", alpha=0.7, color=colors[i]
        )

    ax.axhline(70, color="red", linestyle="--", linewidth=2, label="Warning (70°C)")

    for boundary_time, name in transition_boundaries:
        ax.axvline(boundary_time, color="gray", linestyle=":", alpha=0.5)

    add_pose_markers(ax)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Motor Temperatures")
    ax.legend(ncol=7, fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 6: Position Errors
    ax = axes[5]
    if len(all_errors) > 0 and all_errors.size > 0:
        for i in range(6):
            ax.plot(
                all_times,
                np.rad2deg(all_errors[:, i]),  # Convert to degrees
                label=f"Motor {i+1}",
                alpha=0.7,
                color=colors[i],
            )

        for boundary_time, name in transition_boundaries:
            ax.axvline(boundary_time, color="gray", linestyle=":", alpha=0.5)

        add_pose_markers(ax)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position Error (°)")
        ax.set_title("Position Tracking Errors (Target - Actual)")
        ax.legend(ncol=6, fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No position error data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Position Tracking Errors")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def run_motion_sequence(controller, kinematics, payload_g, output_dir):
    """
    Run through the full pose sequence and track torques during all motions.
    """
    print("\n" + "=" * 100)
    print(f"MOTION SEQUENCE TEST - PAYLOAD: {payload_g}g")
    print("=" * 100)

    # Compute all IK solutions first
    print("\n1. Computing inverse kinematics for all poses...")
    joint_positions_sequence = []

    for pose_dict in TEST_POSE_SEQUENCE:
        try:
            joint_positions = kinematics.ik(pose_dict["pose"], body_yaw=0.0)
            joint_positions_sequence.append(joint_positions)
            print(f"   ✓ {pose_dict['name']}")
        except Exception as e:
            print(f"   ✗ {pose_dict['name']}: {e}")
            return False

    # Move to initial pose
    print("\n2. Moving to initial neutral pose...")
    current_pos = get_current_positions(controller)
    initial_target = joint_positions_sequence[0]

    # Silent move to initial pose
    steps = int(MOTION_DURATION * MOTION_FREQUENCY)
    period = 1.0 / MOTION_FREQUENCY
    motor_ids = [MOTOR_IDS["body_rotation"]] + [
        MOTOR_IDS[f"stewart_{j}"] for j in range(1, 7)
    ]

    for i in range(steps):
        loop_start = time.time()

        alpha = i / steps if steps > 0 else 1.0
        pos = current_pos + alpha * (initial_target - current_pos)

        for j, motor_id in enumerate(motor_ids):
            val = pos[j]
            if isinstance(val, np.ndarray):
                val = val.item()
            controller.write_goal_position(motor_id, float(val))

        # Sleep for remaining time in this period
        elapsed = time.time() - loop_start
        sleep_time = period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("   ✓ Reached initial pose")
    time.sleep(1.0)  # Stabilize

    # Place payload
    is_interactive = sys.stdin.isatty()

    if payload_g > 0:
        print(f"\n3. >>> Place {payload_g}g weight on the robot head <<<")
        if is_interactive:
            input("   Press ENTER when ready...")
        else:
            print("   Auto-proceeding in 2s...")
            time.sleep(2)
    else:
        print(f"\n3. >>> Baseline test (no weight) <<<")
        if is_interactive:
            input("   Press ENTER to start...")
        else:
            print("   Auto-starting in 2s...")
            time.sleep(2)

    # Run through motion sequence
    print("\n4. Running motion sequence...")
    print("=" * 100)

    all_motion_data = []
    transition_names = []
    pose_achievements = [
        (0.0, TEST_POSE_SEQUENCE[0]["name"])
    ]  # Track when poses are achieved
    cumulative_time = 0.0

    monitor = MotionMonitor(controller, MONITORING_FREQUENCY)

    for i in range(len(TEST_POSE_SEQUENCE) - 1):
        current_pose = TEST_POSE_SEQUENCE[i]
        next_pose = TEST_POSE_SEQUENCE[i + 1]

        transition_name = f"{current_pose['key']}_to_{next_pose['key']}"
        transition_names.append(transition_name)

        current_joints = joint_positions_sequence[i]
        next_joints = joint_positions_sequence[i + 1]

        print(f"\n  Transition {i+1}/{len(TEST_POSE_SEQUENCE)-1}:")
        print(f"    From: {current_pose['name']}")
        print(f"    To:   {next_pose['name']}")

        # Move with monitoring
        motion_data = move_with_monitoring(
            controller,
            monitor,
            current_joints,
            next_joints,
            duration=MOTION_DURATION,
            transition_name=transition_name,
        )

        all_motion_data.append(motion_data)

        # Track pose achievement (at end of motion)
        if motion_data and len(motion_data) > 0:
            motion_duration = motion_data[-1]["time"]
            cumulative_time += motion_duration
            pose_achievements.append((cumulative_time, next_pose["name"]))

        # Check for safety violations
        if motion_data:
            max_torque = max(d["max_torque"] for d in motion_data)
            if max_torque >= SAFE_LIMIT_NM:
                print(
                    f"\n  ⚠ WARNING: Torque {max_torque:.3f} N·m exceeded safe limit during motion!"
                )
                if is_interactive:
                    response = input("  Continue? (y/n): ")
                    if response.lower() != "y":
                        return False

    print("\n" + "=" * 100)
    print("  ✓ Motion sequence complete!")

    # Save data
    print("\n5. Saving data...")

    csv_file = output_dir / f"motion_sequence_{payload_g}g.csv"
    save_motion_data_to_csv(all_motion_data, csv_file, payload_g, transition_names)
    print(f"   ✓ Saved CSV: {csv_file.name}")

    graph_file = output_dir / f"motion_sequence_{payload_g}g.png"
    generate_motion_graphs(
        all_motion_data,
        graph_file,
        payload_g,
        transition_names,
        pose_achievements=pose_achievements,
    )
    print(f"   ✓ Saved graphs: {graph_file.name}")

    # Summary statistics
    print("\n6. Summary statistics:")
    total_samples = sum(len(d) for d in all_motion_data)
    all_max_torques = [d["max_torque"] for motion in all_motion_data for d in motion]

    # Collect position error statistics
    all_max_pos_errors = [
        d["max_pos_error"]
        for motion in all_motion_data
        for d in motion
        if d.get("max_pos_error") is not None
    ]
    all_rms_pos_errors = [
        d["rms_pos_error"]
        for motion in all_motion_data
        for d in motion
        if d.get("rms_pos_error") is not None
    ]

    if all_max_torques:
        overall_max = max(all_max_torques)
        overall_avg = np.mean(all_max_torques)
        overall_std = np.std(all_max_torques)

        print(f"   Total samples collected: {total_samples}")
        print(
            f"   Overall max torque: {overall_max:.3f} N·m ({overall_max/STALL_TORQUE_NM*100:.1f}% of stall)"
        )
        print(f"   Overall avg torque: {overall_avg:.3f} N·m")
        print(f"   Overall std torque: {overall_std:.3f} N·m")

    if all_max_pos_errors:
        max_pos_error = max(all_max_pos_errors)
        avg_pos_error = np.mean(all_max_pos_errors)

        print(f"\n   Position tracking errors:")
        print(f"   Max position error: {np.rad2deg(max_pos_error):.3f}°")
        print(f"   Avg position error: {np.rad2deg(avg_pos_error):.3f}°")

        if all_rms_pos_errors:
            avg_rms_error = np.mean(all_rms_pos_errors)
            print(f"   Avg RMS error: {np.rad2deg(avg_rms_error):.3f}°")

    # Remove weight
    if payload_g > 0:
        print(f"\n7. >>> Remove the weight <<<")
        if is_interactive:
            input("   Press ENTER when done...")
        else:
            print("   Auto-proceeding...")
            time.sleep(1)

    return True


def detect_serial_port():
    """Auto-detect serial port."""
    ports = list(serial.tools.list_ports.comports())

    print(f"\nDetected {len(ports)} serial port(s):")
    for p in ports:
        print(f"  - {p.device}: {p.description}")

    if ports:
        for p in ports:
            if "usbmodem" in p.device.lower() or "acm" in p.device.lower():
                print(f"\nSelected: {p.device}")
                return p.device

        print(f"\nNo usbmodem/ACM found, using: {ports[0].device}")
        return ports[0].device

    return "/dev/ttyACM0"


def main():
    """Main routine."""
    print("=" * 100)
    print("TRACK TORQUES DURING MOTION - REAL ROBOT")
    print("=" * 100)

    # Get serial port
    port = detect_serial_port()
    print(f"\nUsing serial port: {port}")

    # Setup motors
    controller = setup_motors(port)
    if controller is None:
        print("\nERROR: Failed to connect to motors")
        return

    # Initialize kinematics
    print("\nInitializing kinematics...")
    kinematics = AnalyticalKinematics(automatic_body_yaw=True)
    print("✓ Kinematics ready")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"real_robot_test_results/motion_tracking_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}\n")

    # Get test configuration
    print("=" * 100)
    print("TEST CONFIGURATION")
    print("=" * 100)

    is_interactive = sys.stdin.isatty()

    print(f"\nPose sequence: {len(TEST_POSE_SEQUENCE)} poses")
    print(f"Transitions: {len(TEST_POSE_SEQUENCE) - 1}")
    print(f"Motion duration per transition: {MOTION_DURATION}s")
    print(f"Monitoring frequency: {MONITORING_FREQUENCY} Hz")

    if is_interactive:
        # Weight configuration
        print("\nPayload weights (g):")
        weights_input = input("Enter comma-separated (e.g., 0,100,200): ").strip()

        try:
            payloads = [int(w.strip()) for w in weights_input.split(",")]
        except:
            print("Invalid, using: 0")
            payloads = [0]

        # Confirm
        total_time = len(payloads) * (len(TEST_POSE_SEQUENCE) - 1) * MOTION_DURATION
        print(f"\nTotal tests: {len(payloads)}")
        print(f"Estimated time: ~{int(total_time/60)} minutes")
        response = input("Proceed? (y/n): ")
        if response.lower() != "y":
            print("Cancelled")
            return
    else:
        print("\nNon-interactive mode: using 0g weight for demo")
        payloads = [0]

    # Display PID configuration
    print("\nPID Configuration:")
    print(
        f"  Default: P={DEFAULT_PID['p_gain']}, I={DEFAULT_PID['i_gain']}, D={DEFAULT_PID['d_gain']}"
    )
    if PAYLOAD_PID_CONFIG:
        print("  Custom configurations:")
        for payload, pid in PAYLOAD_PID_CONFIG.items():
            print(
                f"    {payload}g: P={pid['p_gain']}, I={pid['i_gain']}, D={pid['d_gain']}"
            )

    # Run tests
    print("\n" + "=" * 100)
    print("STARTING MOTION TESTS")
    print("=" * 100)

    try:
        for payload_g in payloads:
            # Configure PID for this payload
            configure_pid_for_payload(controller, payload_g)

            success = run_motion_sequence(controller, kinematics, payload_g, output_dir)

            if not success:
                print("\nStopping tests")
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Disable motors
        print("\n\nDisabling motors...")
        for motor_id in MOTOR_IDS.values():
            try:
                controller.write_torque_enable(motor_id, False)
            except:
                pass
        print("✓ Motors disabled")

    # Summary
    print("\n" + "=" * 100)
    print("COMPLETE")
    print("=" * 100)
    print(f"Results: {output_dir}/")
    print("=" * 100)


if __name__ == "__main__":
    main()
