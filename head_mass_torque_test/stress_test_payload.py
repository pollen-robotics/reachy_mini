#!/usr/bin/env python3
"""
Stress Test Robot with Payload - Find Maximum Safe Payload

Runs motion sequences in a loop until motor temperatures reach safety limits.
Tests different payload masses with configurable PID values to find maximum safe payload.

Usage:
    python stress_test_payload.py
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

# Test pose sequence
TEST_POSE_SEQUENCE = [
    {
        "key": "neutral",
        "name": "Neutral",
        "pose": create_head_pose(roll=0, pitch=0, yaw=0, degrees=True),
    },
    {
        "key": "pitch_up_20",
        "name": "Pitch up 20°",
        "pose": create_head_pose(roll=0, pitch=-20, yaw=0, degrees=True),
    },
    {
        "key": "neutral_2",
        "name": "Neutral",
        "pose": create_head_pose(roll=0, pitch=0, yaw=0, degrees=True),
    },
    {
        "key": "pitch_down_20",
        "name": "Pitch down 20°",
        "pose": create_head_pose(roll=0, pitch=20, yaw=0, degrees=True),
    },
    {
        "key": "neutral_3",
        "name": "Neutral",
        "pose": create_head_pose(roll=0, pitch=0, yaw=0, degrees=True),
    },
    {
        "key": "roll_right_20",
        "name": "Roll right 20°",
        "pose": create_head_pose(roll=20, pitch=0, yaw=0, degrees=True),
    },
    {
        "key": "neutral_4",
        "name": "Neutral",
        "pose": create_head_pose(roll=0, pitch=0, yaw=0, degrees=True),
    },
    {
        "key": "roll_left_20",
        "name": "Roll left 20°",
        "pose": create_head_pose(roll=-20, pitch=0, yaw=0, degrees=True),
    },
    {
        "key": "neutral_5",
        "name": "Neutral",
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
TEMP_WARNING_C = 60  # Warning temperature
TEMP_CRITICAL_C = 70  # Critical temperature - stop test
TEMP_CHECK_INTERVAL = 1.0  # Check temperature every second

# Motion parameters
MOTION_DURATION = 0.25  # seconds per transition
MOTION_FREQUENCY = 50  # Hz
MONITORING_FREQUENCY = 100  # Hz

# PID Configuration per payload
PAYLOAD_PID_CONFIG = {
    0: {"p_gain": 1000, "i_gain": 100, "d_gain": 0},
    50: {"p_gain": 1000, "i_gain": 100, "d_gain": 0},
    100: {"p_gain": 1200, "i_gain": 120, "d_gain": 0},
    150: {"p_gain": 1400, "i_gain": 140, "d_gain": 0},
    200: {"p_gain": 120000, "i_gain": 100, "d_gain": 0},
    250: {"p_gain": 1500, "i_gain": 150, "d_gain": 0},
    300: {"p_gain": 2000, "i_gain": 200, "d_gain": 0},
}

DEFAULT_PID = {"p_gain": 1000, "i_gain": 100, "d_gain": 0}

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
        controller = Xl330PyController(port, 1000000, 1.0)
        time.sleep(0.5)
        print("  ✓ Controller created")

        # Enable torque on all motors
        motor_ids = list(MOTOR_IDS.values())
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


class ContinuousMonitor:
    """Continuous motor monitoring with temperature safety checks."""

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
        self.target_positions = None
        self.stop_requested = False
        self.stop_reason = None

    def start(self):
        """Start monitoring."""
        self.running = True
        self.data = []
        self.start_time = time.time()
        self.stop_requested = False
        self.stop_reason = None
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def request_stop(self, reason):
        """Request stop from monitoring thread."""
        with self.lock:
            self.stop_requested = True
            self.stop_reason = reason

    def is_stop_requested(self):
        """Check if stop was requested."""
        with self.lock:
            return self.stop_requested

    def get_stop_reason(self):
        """Get stop reason."""
        with self.lock:
            return self.stop_reason

    def set_target_positions(self, target_positions):
        """Update target positions for error calculation."""
        with self.lock:
            self.target_positions = np.array(target_positions, dtype=float)

    def _monitor_loop(self):
        """Background monitoring loop."""
        period = 1.0 / self.frequency

        while self.running:
            loop_start = time.time()

            try:
                # Read current, position, and temperature from all stewart motors
                currents = []
                positions = []
                temperatures = []

                for motor_id in self.stewart_ids:
                    current = self.controller.read_present_current(motor_id)
                    pos = self.controller.read_present_position(motor_id)
                    temp = self.controller.read_present_temperature(motor_id)

                    if isinstance(current, (list, np.ndarray)):
                        current = float(current[0])
                    if isinstance(pos, (list, np.ndarray)):
                        pos = float(pos[0])
                    if isinstance(temp, (list, np.ndarray)):
                        temp = float(temp[0])

                    currents.append(current)
                    positions.append(pos)
                    temperatures.append(temp)

                # Check temperature limits
                max_temp = max(temperatures)
                if max_temp >= TEMP_CRITICAL_C:
                    self.request_stop(
                        f"Critical temperature: {max_temp:.0f}°C (motor {temperatures.index(max_temp)+1})"
                    )
                    self.running = False
                    break

                elapsed = time.time() - self.start_time
                torques = [current_to_torque(c) for c in currents]
                max_abs_current = max(abs(c) for c in currents)
                max_abs_torque = max(abs(t) for t in torques)

                # Calculate position errors
                with self.lock:
                    position_errors = None
                    target_pos = None
                    max_pos_error = None
                    rms_pos_error = None

                    if self.target_positions is not None:
                        target_pos = self.target_positions[1:].tolist()
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


def move_with_monitoring(controller, monitor, current_pos, target_pos, duration=0.5):
    """Move motors smoothly while monitoring."""
    current_pos = np.array(current_pos, dtype=float)
    target_pos = np.array(target_pos, dtype=float)

    motor_ids = [MOTOR_IDS["body_rotation"]] + [
        MOTOR_IDS[f"stewart_{j}"] for j in range(1, 7)
    ]

    steps = int(duration * MOTION_FREQUENCY)
    period = 1.0 / MOTION_FREQUENCY

    for i in range(steps):
        loop_start = time.time()

        # Check if stop was requested
        if monitor.is_stop_requested():
            return False

        alpha = i / steps if steps > 0 else 1.0
        pos = current_pos + alpha * (target_pos - current_pos)

        # Update target in monitor
        monitor.set_target_positions(pos)

        # Set positions
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

    return True


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


def save_stress_test_data(all_data, csv_file, payload_g, cycle_count):
    """Save all monitoring data to CSV."""
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "payload_g",
                "cycle",
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

        for sample in all_data:
            target_positions = sample.get("target_positions") or [None] * 6
            position_errors_rad = sample.get("position_errors") or [None] * 6
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
                    payload_g,
                    cycle_count,
                    sample["time"],
                    *sample["currents"],
                    *sample["positions"],
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


def generate_stress_test_graphs(
    all_data, output_file, payload_g, cycle_count, stop_reason
):
    """Generate graphs for stress test data."""
    if not all_data:
        return

    times = np.array([d["time"] for d in all_data])
    currents = np.array([d["currents"] for d in all_data])
    torques = np.array([d["torques"] for d in all_data])
    temps = np.array([d["temperatures"] for d in all_data])
    max_temps = [d["max_temp"] for d in all_data]
    max_torques = [d["max_torque"] for d in all_data]

    # Position errors
    errors_data = [
        d.get("position_errors")
        for d in all_data
        if d.get("position_errors") is not None
    ]
    if errors_data:
        errors = np.array(errors_data)
    else:
        errors = None

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    colors = plt.cm.tab10(np.linspace(0, 1, 6))

    # Plot 1: Motor Temperatures
    ax = axes[0]
    for i in range(6):
        ax.plot(times, temps[:, i], label=f"Motor {i+1}", alpha=0.7, color=colors[i])

    ax.axhline(
        TEMP_WARNING_C,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Warning ({TEMP_WARNING_C}°C)",
    )
    ax.axhline(
        TEMP_CRITICAL_C,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Critical ({TEMP_CRITICAL_C}°C)",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(
        f"Motor Temperatures - {payload_g}g Payload - {cycle_count} Cycles\nStop reason: {stop_reason}"
    )
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 2: Motor Torques
    ax = axes[1]
    for i in range(6):
        ax.plot(times, torques[:, i], label=f"Motor {i+1}", alpha=0.7, color=colors[i])

    ax.axhline(
        SAFE_LIMIT_NM, color="red", linestyle="--", linewidth=2, label="Safe Limit"
    )
    ax.axhline(-SAFE_LIMIT_NM, color="red", linestyle="--", linewidth=2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (N·m)")
    ax.set_title(f"Motor Torques Over Time")
    ax.legend(ncol=7, fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 3: Motor Currents
    ax = axes[2]
    for i in range(6):
        ax.plot(times, currents[:, i], label=f"Motor {i+1}", alpha=0.7, color=colors[i])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (mA)")
    ax.set_title("Motor Currents Over Time")
    ax.legend(ncol=6, fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 4: Position Errors (if available)
    ax = axes[3]
    if errors is not None and len(errors) > 0:
        error_times = np.array(
            [d["time"] for d in all_data if d.get("position_errors") is not None]
        )
        for i in range(6):
            ax.plot(
                error_times,
                np.rad2deg(errors[:, i]),
                label=f"Motor {i+1}",
                alpha=0.7,
                color=colors[i],
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position Error (°)")
        ax.set_title("Position Tracking Errors")
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
    print("STRESS TEST - FIND MAXIMUM SAFE PAYLOAD")
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
    output_dir = Path(f"stress_test_results/test_{timestamp}")
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
    print(
        f"Temperature limits: Warning={TEMP_WARNING_C}°C, Critical={TEMP_CRITICAL_C}°C"
    )

    print("\nAvailable PID configurations:")
    for payload_g, pid in sorted(PAYLOAD_PID_CONFIG.items()):
        print(
            f"  {payload_g}g: P={pid['p_gain']}, I={pid['i_gain']}, D={pid['d_gain']}"
        )

    if is_interactive:
        payload_input = input("\nPayload weight (g): ").strip()
        try:
            payload_g = int(payload_input)
        except:
            print("Invalid, using: 0")
            payload_g = 0

        print(f"\nPayload: {payload_g}g")
        print(f">>> Place {payload_g}g weight on the robot head <<<")
        input("Press ENTER when ready...")
    else:
        print("\nNon-interactive mode: using 0g weight for demo")
        payload_g = 0

    # Configure PID
    configure_pid_for_payload(controller, payload_g)

    # Compute IK solutions
    print("\nComputing inverse kinematics...")
    joint_positions_sequence = []

    for pose_dict in TEST_POSE_SEQUENCE:
        try:
            joint_positions = kinematics.ik(pose_dict["pose"], body_yaw=0.0)
            joint_positions_sequence.append(joint_positions)
            print(f"   ✓ {pose_dict['name']}")
        except Exception as e:
            print(f"   ✗ {pose_dict['name']}: {e}")
            return

    # Move to initial pose
    print("\nMoving to initial neutral pose...")
    current_pos = get_current_positions(controller)
    initial_target = joint_positions_sequence[0]

    motor_ids = [MOTOR_IDS["body_rotation"]] + [
        MOTOR_IDS[f"stewart_{j}"] for j in range(1, 7)
    ]
    steps = int(MOTION_DURATION * MOTION_FREQUENCY)
    period = 1.0 / MOTION_FREQUENCY

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
    time.sleep(1.0)

    # Start stress test
    print("\n" + "=" * 100)
    print("STARTING STRESS TEST - RUNNING UNTIL TEMPERATURE LIMIT")
    print("=" * 100)
    print(f"Press Ctrl+C to stop manually\n")

    monitor = ContinuousMonitor(controller, MONITORING_FREQUENCY)
    monitor.start()

    cycle_count = 0
    try:
        while True:
            cycle_count += 1
            print(f"\n{'='*100}")
            print(f"CYCLE {cycle_count}")
            print(f"{'='*100}")

            # Run through pose sequence
            for i in range(len(TEST_POSE_SEQUENCE) - 1):
                current_pose = TEST_POSE_SEQUENCE[i]
                next_pose = TEST_POSE_SEQUENCE[i + 1]

                current_joints = joint_positions_sequence[i]
                next_joints = joint_positions_sequence[i + 1]

                # Display status
                latest = monitor.get_latest()
                if latest:
                    print(
                        f"  {current_pose['name']} → {next_pose['name']} | "
                        f"τ_max: {latest['max_torque']:.3f} N·m | "
                        f"T_max: {latest['max_temp']:.0f}°C",
                        end="\r",
                    )

                # Move with monitoring
                success = move_with_monitoring(
                    controller, monitor, current_joints, next_joints, MOTION_DURATION
                )

                if not success:
                    break

            # Check if stop was requested
            if monitor.is_stop_requested():
                break

            # Brief check after each cycle
            latest = monitor.get_latest()
            if latest:
                print(
                    f"\n  Cycle {cycle_count} complete | "
                    f"Max temp: {latest['max_temp']:.0f}°C | "
                    f"Max torque: {latest['max_torque']:.3f} N·m"
                )

    except KeyboardInterrupt:
        print("\n\nManually stopped by user")
        monitor.request_stop("User interrupt (Ctrl+C)")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback

        traceback.print_exc()
        monitor.request_stop(f"Error: {str(e)}")
    finally:
        # Stop monitoring
        monitor.stop()
        all_data = monitor.get_all_data()
        stop_reason = monitor.get_stop_reason() or "Test completed normally"

        print("\n" + "=" * 100)
        print("TEST STOPPED")
        print("=" * 100)
        print(f"Reason: {stop_reason}")
        print(f"Cycles completed: {cycle_count}")
        print(f"Total samples: {len(all_data)}")

        # Save data
        print("\nSaving data...")
        csv_file = output_dir / f"stress_test_{payload_g}g.csv"
        save_stress_test_data(all_data, csv_file, payload_g, cycle_count)
        print(f"   ✓ Saved CSV: {csv_file.name}")

        graph_file = output_dir / f"stress_test_{payload_g}g.png"
        generate_stress_test_graphs(
            all_data, graph_file, payload_g, cycle_count, stop_reason
        )
        print(f"   ✓ Saved graphs: {graph_file.name}")

        # Statistics
        if all_data:
            max_temp_overall = max(d["max_temp"] for d in all_data)
            max_torque_overall = max(d["max_torque"] for d in all_data)
            avg_torque = np.mean([d["max_torque"] for d in all_data])

            print(f"\nStatistics:")
            print(f"   Max temperature: {max_temp_overall:.0f}°C")
            print(
                f"   Max torque: {max_torque_overall:.3f} N·m ({max_torque_overall/STALL_TORQUE_NM*100:.1f}% of stall)"
            )
            print(f"   Avg torque: {avg_torque:.3f} N·m")

        # Disable motors
        print("\nDisabling motors...")
        for motor_id in MOTOR_IDS.values():
            try:
                controller.write_torque_enable(motor_id, False)
            except:
                pass
        print("✓ Motors disabled")

        print("\n" + "=" * 100)
        print("COMPLETE")
        print("=" * 100)
        print(f"Results: {output_dir}/")
        print("=" * 100)


if __name__ == "__main__":
    main()
