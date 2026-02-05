#!/usr/bin/env python3
"""
Test Head Poses on Real Reachy Mini - Direct Motor Control

Uses analytical kinematics + rustypot for direct motor control and current reading.
No daemon required.

Usage:
    python test_real_robot_poses.py
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

# Test poses
TEST_POSES = {
    "neutral": {
        "name": "Neutral (0°, 0°)",
        "pose": create_head_pose(roll=0, pitch=0, yaw=0, degrees=True),
    },
    "pitch_up_20": {
        "name": "Pitch up 20°",
        "pose": create_head_pose(roll=0, pitch=-20, yaw=0, degrees=True),
    },
    "pitch_down_20": {
        "name": "Pitch down 20°",
        "pose": create_head_pose(roll=0, pitch=20, yaw=0, degrees=True),
    },
    "roll_right_20": {
        "name": "Roll right 20°",
        "pose": create_head_pose(roll=20, pitch=0, yaw=0, degrees=True),
    },
    "roll_left_20": {
        "name": "Roll left 20°",
        "pose": create_head_pose(roll=-20, pitch=0, yaw=0, degrees=True),
    },
}

# Torque constants
K_NM_TO_MA = 1.47 / 0.52 * 1000
CORRECTION_FACTOR = 3.0
EFFICIENCY = 1.0

# Safety limits
STALL_TORQUE_NM = 0.6
SAFE_LIMIT_NM = 0.45

# Monitoring
MONITORING_FREQUENCY = 10  # Hz

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
        # Create controller - Xl330PyController(port, baudrate, timeout)
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


def move_to_position(controller, joint_positions, duration=3.0):
    """Move motors to target positions smoothly."""
    # Get current positions
    current_pos = []
    motor_ids = [MOTOR_IDS["body_rotation"]] + [
        MOTOR_IDS[f"stewart_{j}"] for j in range(1, 7)
    ]

    print(f"    Reading current positions...")
    for motor_id in motor_ids:
        pos = controller.read_present_position(motor_id)
        # Position comes as array, extract scalar
        if isinstance(pos, (list, np.ndarray)):
            pos = float(pos[0])
        current_pos.append(pos)
        print(f"      Motor {motor_id}: {pos:.4f}")

    current_pos = np.array(current_pos, dtype=float)
    target_pos = np.array(joint_positions, dtype=float)

    print(f"    Current shape: {current_pos.shape}, Target shape: {target_pos.shape}")
    print(f"    Interpolating over {duration}s...")

    # Interpolate
    steps = int(duration * 50)  # 50 Hz
    for i in range(steps + 1):
        alpha = i / steps
        pos = current_pos + alpha * (target_pos - current_pos)

        # Set positions
        for j, motor_id in enumerate(motor_ids):
            val = pos[j]
            if isinstance(val, np.ndarray):
                val = val.item()
            controller.write_goal_position(motor_id, float(val))

        if i % 25 == 0:
            print(f"      Progress: {i}/{steps}", end="\r")

        time.sleep(1.0 / 50)

    print(f"      Progress: {steps}/{steps} - Done!")


class MotorMonitor:
    """Continuous motor monitoring in background thread."""

    def __init__(self, controller, frequency=10):
        self.controller = controller
        self.frequency = frequency
        self.running = False
        self.thread = None
        self.data = []
        self.start_time = None
        self.stewart_ids = [MOTOR_IDS[f"stewart_{i}"] for i in range(1, 7)]

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

    def _monitor_loop(self):
        """Background monitoring loop."""
        period = 1.0 / self.frequency

        while self.running:
            loop_start = time.time()

            try:
                # Read current and temperature from all stewart motors
                currents = []
                temperatures = []

                for motor_id in self.stewart_ids:
                    current = self.controller.read_present_current(motor_id)
                    temp = self.controller.read_present_temperature(motor_id)

                    # Extract scalar from array if needed
                    if isinstance(current, (list, np.ndarray)):
                        current = float(current[0])
                    if isinstance(temp, (list, np.ndarray)):
                        temp = float(temp[0])

                    currents.append(current)
                    temperatures.append(temp)

                elapsed = time.time() - self.start_time
                torques = [current_to_torque(c) for c in currents]
                max_abs_current = max(abs(c) for c in currents)
                max_abs_torque = max(abs(t) for t in torques)
                max_temp = max(temperatures)

                self.data.append(
                    {
                        "time": elapsed,
                        "currents": currents,
                        "temperatures": temperatures,
                        "torques": torques,
                        "max_current": max_abs_current,
                        "max_torque": max_abs_torque,
                        "max_temp": max_temp,
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
        if self.data:
            return self.data[-1]
        return None

    def get_all_data(self):
        """Get all collected data."""
        return self.data.copy()


def save_data_to_csv(data, csv_file, pose_key, pose_name, payload_g):
    """Save monitoring data to CSV."""
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(
            [
                "pose_key",
                "pose_name",
                "payload_g",
                "time_s",
                "current_m1_ma",
                "current_m2_ma",
                "current_m3_ma",
                "current_m4_ma",
                "current_m5_ma",
                "current_m6_ma",
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
            ]
        )

        # Data rows
        for sample in data:
            writer.writerow(
                [
                    pose_key,
                    pose_name,
                    payload_g,
                    sample["time"],
                    *sample["currents"],
                    *sample["temperatures"],
                    *sample["torques"],
                    sample["max_current"],
                    sample["max_torque"],
                    sample["max_temp"],
                ]
            )


def generate_graph(data, output_file, pose_name, payload_g):
    """Generate graphs for the collected data."""
    if not data:
        return

    times = [d["time"] for d in data]
    currents = np.array([d["currents"] for d in data])
    torques = np.array([d["torques"] for d in data])
    temps = np.array([d["temperatures"] for d in data])
    max_currents = [d["max_current"] for d in data]
    max_torques = [d["max_torque"] for d in data]
    max_temps = [d["max_temp"] for d in data]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Plot 1: Individual motor currents
    ax = axes[0]
    for i in range(6):
        ax.plot(times, currents[:, i], label=f"Motor {i+1}", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (mA)")
    ax.set_title(f"Motor Currents - {pose_name} ({payload_g}g)")
    ax.legend(ncol=6)
    ax.grid(alpha=0.3)

    # Plot 2: Individual motor torques
    ax = axes[1]
    for i in range(6):
        ax.plot(times, torques[:, i], label=f"Motor {i+1}", alpha=0.7)
    ax.axhline(
        SAFE_LIMIT_NM, color="red", linestyle="--", linewidth=2, label="Safe Limit"
    )
    ax.axhline(-SAFE_LIMIT_NM, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (N·m)")
    ax.set_title(f"Motor Torques - {pose_name} ({payload_g}g)")
    ax.legend(ncol=6)
    ax.grid(alpha=0.3)

    # Plot 3: Temperatures
    ax = axes[2]
    for i in range(6):
        ax.plot(times, temps[:, i], label=f"Motor {i+1}", alpha=0.7)
    ax.axhline(70, color="red", linestyle="--", linewidth=2, label="Warning (70°C)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Motor Temperatures - {pose_name} ({payload_g}g)")
    ax.legend(ncol=6)
    ax.grid(alpha=0.3)

    # Plot 4: Maximum values
    ax = axes[3]
    ax.plot(times, max_currents, label="Max Current (mA)", color="blue", linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(times, max_torques, label="Max Torque (N·m)", color="red", linewidth=2)
    ax2.plot(times, max_temps, label="Max Temp (°C)", color="orange", linewidth=2)
    ax2.axhline(SAFE_LIMIT_NM, color="darkred", linestyle="--", linewidth=2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Max Current (mA)", color="blue")
    ax2.set_ylabel("Max Torque (N·m) / Temp (°C)", color="red")
    ax.set_title(f"Maximum Values - {pose_name} ({payload_g}g)")
    ax.grid(alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def display_live_data(monitor):
    """Display live monitoring data in terminal."""
    latest = monitor.get_latest()

    if latest is None:
        print("Waiting for data...", end="\r")
        return

    currents = latest["currents"]
    temps = latest["temperatures"]
    max_current = latest["max_current"]
    max_torque = latest["max_torque"]
    max_temp = latest["max_temp"]

    # Clear line
    print(" " * 120, end="\r")

    # Status indicator
    if max_torque >= SAFE_LIMIT_NM or max_temp >= 70:
        status = "⚠ DANGER"
        color = "\033[91m"  # Red
    elif max_torque >= SAFE_LIMIT_NM * 0.8 or max_temp >= 60:
        status = "⚡ WARNING"
        color = "\033[93m"  # Yellow
    else:
        status = "✓ OK"
        color = "\033[92m"  # Green

    reset = "\033[0m"

    # Format current values
    curr_str = " ".join([f"M{i+1}:{c:5.0f}" for i, c in enumerate(currents)])

    # Print
    print(
        f"{color}{status}{reset} | t:{latest['time']:5.1f}s | I(mA): {curr_str} | Max:{max_current:5.0f} | τ:{max_torque:.3f}N·m | T:{max_temp:.0f}°C",
        end="\r",
    )


def test_pose_with_payload(
    controller, kinematics, monitor, pose_key, pose_dict, payload_g, output_dir
):
    """Test a single pose with payload."""
    pose_name = pose_dict["name"]
    pose = pose_dict["pose"]

    print("\n" + "=" * 100)
    print(f"TESTING: {pose_name} | PAYLOAD: {payload_g}g")
    print("=" * 100)

    # Compute IK
    print(f"\n1. Computing inverse kinematics...")
    try:
        joint_positions = kinematics.ik(pose, body_yaw=0.0)
        print(f"   ✓ Joint positions (rad): {joint_positions}")
        print(f"   ✓ Joint positions (deg): {np.rad2deg(joint_positions)}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Move to pose
    print(f"\n2. Moving to pose...")
    try:
        move_to_position(controller, joint_positions, duration=3.0)
        print("   ✓ Reached pose")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Stabilize
    print(f"\n3. Stabilizing (2s)...")
    time.sleep(2.0)
    print("   ✓ Stabilized")

    # Add weight
    is_interactive = sys.stdin.isatty()

    if payload_g > 0:
        print(f"\n4. >>> Place {payload_g}g weight on the robot head <<<")
        if is_interactive:
            input("   Press ENTER when ready...")
        else:
            print("   Auto-proceeding in 2s...")
            time.sleep(2)
    else:
        print(f"\n4. >>> Baseline test (no weight) <<<")
        if is_interactive:
            input("   Press ENTER to start...")
        else:
            print("   Auto-starting in 2s...")
            time.sleep(2)

    # Start monitoring
    is_interactive = sys.stdin.isatty()
    monitor_duration = 5.0  # seconds for non-interactive mode

    print(f"\n5. Monitoring motors...")
    if is_interactive:
        print("   (press ENTER to stop)")
    else:
        print(f"   (auto-stopping after {monitor_duration}s)")
    print("=" * 100)

    monitor.start()

    # Live display loop
    if is_interactive:
        # Interactive mode - use non-blocking input check on main thread (macOS compatible)
        while True:
            display_live_data(monitor)

            # Non-blocking check for Enter key
            # Check if stdin has data available without blocking
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                break

            time.sleep(0.1)
    else:
        # Auto mode - monitor for fixed duration
        start_time = time.time()
        while (time.time() - start_time) < monitor_duration:
            display_live_data(monitor)
            time.sleep(0.1)

    print("\n" + "=" * 100)

    # Stop monitoring
    monitor.stop()
    data = monitor.get_all_data()

    print(f"   ✓ Collected {len(data)} samples")

    # Check for errors
    if data:
        max_torque = max(d["max_torque"] for d in data)
        max_temp = max(d["max_temp"] for d in data)

        if max_torque >= SAFE_LIMIT_NM:
            print(f"   ⚠ WARNING: Max torque {max_torque:.3f} N·m exceeded safe limit!")

        if max_temp >= 70:
            print(f"   ⚠ WARNING: Max temperature {max_temp:.0f}°C is high!")

        if max_torque >= SAFE_LIMIT_NM or max_temp >= 70:
            response = input("   Continue? (y/n): ")
            if response.lower() != "y":
                return False

    # Save data
    csv_file = output_dir / f"{pose_key}_{payload_g}g.csv"
    save_data_to_csv(data, csv_file, pose_key, pose_name, payload_g)
    print(f"   ✓ Saved CSV: {csv_file.name}")

    # Generate graph
    graph_file = output_dir / f"{pose_key}_{payload_g}g.png"
    generate_graph(data, graph_file, pose_name, payload_g)
    print(f"   ✓ Saved graph: {graph_file.name}")

    # Remove weight
    is_interactive = sys.stdin.isatty()

    if payload_g > 0:
        print(f"\n6. >>> Remove the weight <<<")
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
        # Look for usbmodem (macOS) or ACM (Linux)
        for p in ports:
            if "usbmodem" in p.device.lower() or "acm" in p.device.lower():
                print(f"\nSelected: {p.device}")
                return p.device

        # If no usbmodem/ACM, return first port
        print(f"\nNo usbmodem/ACM found, using: {ports[0].device}")
        return ports[0].device

    return "/dev/ttyACM0"


def main():
    """Main routine."""
    print("=" * 100)
    print("REAL ROBOT HEAD POSE TEST - DIRECT MOTOR CONTROL")
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
    output_dir = Path(f"real_robot_test_results/test_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}\n")

    # Get test configuration
    print("=" * 100)
    print("TEST CONFIGURATION")
    print("=" * 100)

    # Check if running interactively
    is_interactive = sys.stdin.isatty()

    # Pose selection
    print("\nAvailable poses:")
    pose_keys = list(TEST_POSES.keys())
    for i, key in enumerate(pose_keys, 1):
        print(f"  {i}. {TEST_POSES[key]['name']}")

    if is_interactive:
        choice = input("\nTest all or select? (1=all, 2=select): ").strip()

        if choice == "2":
            selected_indices = input("Enter pose numbers (e.g., 1,3,5): ").strip()
            try:
                selected = [int(i.strip()) - 1 for i in selected_indices.split(",")]
                test_poses = {
                    pose_keys[i]: TEST_POSES[pose_keys[i]]
                    for i in selected
                    if 0 <= i < len(pose_keys)
                }
            except:
                print("Invalid, using all poses")
                test_poses = TEST_POSES
        else:
            test_poses = TEST_POSES

        # Weight configuration
        print("\nPayload weights (g):")
        weights_input = input("Enter comma-separated (e.g., 0,100,200): ").strip()

        try:
            payloads = [int(w.strip()) for w in weights_input.split(",")]
        except:
            print("Invalid, using: 0,100")
            payloads = [0, 100]

        # Confirm
        print(f"\nTotal tests: {len(test_poses) * len(payloads)}")
        response = input("Proceed? (y/n): ")
        if response.lower() != "y":
            print("Cancelled")
            return
    else:
        # Non-interactive mode: use defaults
        print("\nNon-interactive mode: using first pose and 0g weight for demo")
        test_poses = {"neutral": TEST_POSES["neutral"]}
        payloads = [0]

    # Run tests
    print("\n" + "=" * 100)
    print("STARTING TESTS")
    print("=" * 100)

    monitor = MotorMonitor(controller, MONITORING_FREQUENCY)

    try:
        for payload_g in payloads:
            print(f"\n\n{'#' * 100}")
            print(f"PAYLOAD: {payload_g}g")
            print(f"{'#' * 100}")

            for pose_key, pose_dict in test_poses.items():
                success = test_pose_with_payload(
                    controller,
                    kinematics,
                    monitor,
                    pose_key,
                    pose_dict,
                    payload_g,
                    output_dir,
                )

                if not success:
                    print("\nStopping")
                    break

            if not success:
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted")
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
