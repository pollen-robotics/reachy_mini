#!/usr/bin/env python3
"""
Track Motor Torques During Motion Between Poses - MuJoCo Simulation

Simulated version that tracks torques continuously during motion/transition
between poses using MuJoCo physics simulation.

Usage:
    python track_motion_torques_sim.py
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import csv
import matplotlib.pyplot as plt
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reachy_mini.daemon.backend.mujoco.backend import MujocoBackend
from reachy_mini.utils import create_head_pose

# ============================================================================
# CONFIGURATION
# ============================================================================

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

SCENE = "empty_payload"

# Torque reference
STALL_TORQUE_NM = 0.6  # Motor stall torque
SAFE_LIMIT_NM = 0.45  # 75% of stall torque

# Motion parameters
MOTION_DURATION = 0.25  # seconds per transition (matches real robot)
SIMULATION_FREQUENCY = 500  # Hz - MuJoCo simulation frequency
CONTROL_FREQUENCY = 50  # Hz - control update frequency (match real robot)
SAMPLING_FREQUENCY = 100  # Hz - data collection frequency

# Payload mass testing
PAYLOAD_MASSES_G = [0, 50, 100, 150, 200, 300]  # Payload masses to test in grams

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def render_robot_pose(backend, output_path, width=400, height=400):
    """Render current robot state and save image."""
    renderer = mujoco.Renderer(backend.model, height=height, width=width)
    renderer.update_scene(backend.data)
    pixels = renderer.render()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(pixels)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()

    return pixels


def modify_payload_mass(model, mass_kg):
    """
    Modify the fake_payload_head mass in the MuJoCo model.

    Args:
        model: MuJoCo model object
        mass_kg: New payload mass value in kg

    Returns:
        original_mass_kg: The original payload mass value
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fake_payload_head")

    # Get original mass
    original_mass = model.body_mass[body_id]

    # Set new mass
    model.body_mass[body_id] = mass_kg

    return original_mass


def simulate_motion_with_tracking(
    backend, current_joints, target_joints, duration, transition_name, render_path=None
):
    """
    Simulate smooth motion from current to target joints while tracking torques.

    Args:
        backend: MuJoCo backend
        current_joints: Starting joint positions
        target_joints: Target joint positions
        duration: Duration of motion in seconds
        transition_name: Name of the transition
        render_path: Optional path to save render image when pose is achieved

    Returns:
        Tuple of (motion_data, achieved_time) where achieved_time is when target pose was reached
    """
    print(f"\n  Starting transition: {transition_name}")
    print(f"    Duration: {duration}s")

    current_joints = np.array(current_joints, dtype=float)
    target_joints = np.array(target_joints, dtype=float)

    # Calculate number of steps
    total_sim_steps = int(duration * SIMULATION_FREQUENCY)
    total_control_steps = int(duration * CONTROL_FREQUENCY)
    control_interval = int(SIMULATION_FREQUENCY / CONTROL_FREQUENCY)
    sampling_interval = int(SIMULATION_FREQUENCY / SAMPLING_FREQUENCY)

    motion_data = []
    start_time = backend.data.time

    # Pre-compute control trajectory at CONTROL_FREQUENCY
    control_trajectory = []
    for i in range(total_control_steps + 1):
        alpha = i / total_control_steps if total_control_steps > 0 else 1.0
        interpolated_joints = current_joints + alpha * (target_joints - current_joints)
        control_trajectory.append(interpolated_joints)

    for step in range(total_sim_steps + 1):
        # Update control target at CONTROL_FREQUENCY (e.g., every 10 sim steps for 50 Hz)
        control_idx = min(step // control_interval, len(control_trajectory) - 1)
        backend.data.ctrl[:7] = control_trajectory[control_idx]

        # Step simulation
        mujoco.mj_step(backend.model, backend.data)

        # Collect data at sampling frequency
        if step % sampling_interval == 0:
            # Read torques from actuators (indices 1-6 are stewart motors, 0 is body rotation)
            stewart_torques = [
                float(backend.data.actuator_force[i]) for i in range(1, 7)
            ]

            # Get current positions (extract scalars from joint positions)
            stewart_positions = []
            for i in range(1, 7):  # Skip body rotation at index 0
                qpos_addr = backend.joint_qpos_addr[i]
                pos = float(backend.data.qpos[qpos_addr])
                stewart_positions.append(pos)

            # Calculate position errors
            target_stewart = [float(interpolated_joints[i]) for i in range(1, 7)]
            position_errors = [
                target_stewart[i] - stewart_positions[i] for i in range(6)
            ]

            # Calculate statistics
            abs_torques = [abs(t) for t in stewart_torques]
            max_torque = max(abs_torques)
            rms_torque = float(np.sqrt(np.mean(np.array(stewart_torques) ** 2)))

            abs_errors = [abs(e) for e in position_errors]
            max_pos_error = max(abs_errors)
            rms_pos_error = float(np.sqrt(np.mean(np.array(position_errors) ** 2)))

            elapsed = backend.data.time - start_time

            sample = {
                "time": elapsed,
                "positions": stewart_positions,
                "target_positions": target_stewart,
                "position_errors": position_errors,
                "torques": stewart_torques,
                "max_torque": max_torque,
                "rms_torque": rms_torque,
                "max_pos_error": max_pos_error,
                "rms_pos_error": rms_pos_error,
            }

            motion_data.append(sample)

        # Progress display
        if step % 100 == 0:
            progress_pct = (
                (step / total_sim_steps) * 100 if total_sim_steps > 0 else 100
            )
            if motion_data:
                latest = motion_data[-1]
                print(
                    f"      Progress: {progress_pct:.0f}% | τ_max: {latest['max_torque']:.3f} N·m | "
                    f"err: {np.rad2deg(latest['max_pos_error']):.3f}°",
                    end="\r",
                )

    print(f"      Progress: 100% - Done!                              ")
    print(f"    ✓ Collected {len(motion_data)} samples during motion")

    # Capture render at achieved pose
    achieved_time = motion_data[-1]["time"] if motion_data else duration
    if render_path:
        render_robot_pose(backend, render_path)
        print(f"    ✓ Saved render: {Path(render_path).name}")

    if motion_data:
        max_torque = max(d["max_torque"] for d in motion_data)
        avg_torque = np.mean([d["max_torque"] for d in motion_data])
        print(f"    ✓ Torque: max={max_torque:.3f} N·m, avg={avg_torque:.3f} N·m")

        max_pos_err = max(d["max_pos_error"] for d in motion_data)
        avg_pos_err = np.mean([d["max_pos_error"] for d in motion_data])
        print(
            f"    ✓ Position error: max={np.rad2deg(max_pos_err):.3f}°, "
            f"avg={np.rad2deg(avg_pos_err):.3f}°"
        )

    return motion_data, achieved_time


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
                "torque_m1_nm",
                "torque_m2_nm",
                "torque_m3_nm",
                "torque_m4_nm",
                "torque_m5_nm",
                "torque_m6_nm",
                "max_torque_nm",
                "rms_torque_nm",
                "max_pos_error_deg",
                "rms_pos_error_deg",
            ]
        )

        # Data rows for each transition
        for transition_idx, (transition_name, data) in enumerate(
            zip(transition_names, all_motion_data)
        ):
            for sample in data:
                # Convert errors to degrees
                position_errors_deg = [
                    np.rad2deg(err) for err in sample["position_errors"]
                ]

                max_pos_error_deg = np.rad2deg(sample["max_pos_error"])
                rms_pos_error_deg = np.rad2deg(sample["rms_pos_error"])

                writer.writerow(
                    [
                        transition_name,
                        payload_g,
                        sample["time"],
                        *sample["positions"],
                        *sample["target_positions"],
                        *position_errors_deg,
                        *sample["torques"],
                        sample["max_torque"],
                        sample["rms_torque"],
                        max_pos_error_deg,
                        rms_pos_error_deg,
                    ]
                )


def generate_motion_graphs(
    all_motion_data,
    output_file,
    payload_g,
    transition_names,
    pose_achievements=None,
    render_dir=None,
):
    """
    Generate comprehensive graphs for motion data.

    Args:
        all_motion_data: List of motion data for each transition
        output_file: Path to save the graph
        payload_g: Payload mass in grams
        transition_names: Names of each transition
        pose_achievements: List of (time, pose_name) tuples indicating when poses were achieved
        render_dir: Directory containing pose render images
    """
    if not all_motion_data or not any(all_motion_data):
        return

    # Combine all data with transition markers
    fig, axes = plt.subplots(5, 1, figsize=(16, 16))

    # Prepare data arrays
    all_times = []
    all_torques = []
    all_positions = []
    all_targets = []
    all_errors = []
    transition_boundaries = []

    current_time_offset = 0

    for transition_idx, (data, name) in enumerate(
        zip(all_motion_data, transition_names)
    ):
        if not data:
            continue

        times = np.array([d["time"] for d in data]) + current_time_offset
        torques = np.array([d["torques"] for d in data])
        positions = np.array([d["positions"] for d in data])
        targets = np.array([d["target_positions"] for d in data])
        errors = np.array([d["position_errors"] for d in data])

        all_times.append(times)
        all_torques.append(torques)
        all_positions.append(positions)
        all_targets.append(targets)
        all_errors.append(errors)

        # Mark transition boundary
        if len(times) > 0:
            transition_boundaries.append((times[-1], name))
            current_time_offset = times[-1]

    # Concatenate all arrays
    all_times = np.concatenate(all_times) if all_times else np.array([])
    all_torques = np.vstack(all_torques) if all_torques else np.array([])
    all_positions = np.vstack(all_positions) if all_positions else np.array([])
    all_targets = np.vstack(all_targets) if all_targets else np.array([])
    all_errors = np.vstack(all_errors) if all_errors else np.array([])

    if len(all_times) == 0:
        return

    colors = plt.cm.tab10(np.linspace(0, 1, 6))

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

    for boundary_time, name in transition_boundaries:
        ax.axvline(boundary_time, color="gray", linestyle=":", alpha=0.5)

    add_pose_markers(ax)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Max Torque (N·m)")
    ax.set_title("Maximum Torque Envelope")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Motor Positions vs Targets
    ax = axes[2]
    for i in range(6):
        ax.plot(
            all_times,
            all_positions[:, i],
            label=f"M{i+1} actual",
            alpha=0.7,
            color=colors[i],
            linestyle="-",
        )
        ax.plot(
            all_times,
            all_targets[:, i],
            alpha=0.4,
            color=colors[i],
            linestyle="--",
            linewidth=1,
        )

    for boundary_time, name in transition_boundaries:
        ax.axvline(boundary_time, color="gray", linestyle=":", alpha=0.5)

    add_pose_markers(ax)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (rad)")
    ax.set_title("Motor Positions (solid) vs Targets (dashed)")
    ax.legend(ncol=6, fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 4: Position Errors
    ax = axes[3]
    for i in range(6):
        ax.plot(
            all_times,
            np.rad2deg(all_errors[:, i]),
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

    # Plot 5: RMS values over time
    ax = axes[4]
    rms_torques = [d["rms_torque"] for motion in all_motion_data for d in motion]
    rms_errors = [d["rms_pos_error"] for motion in all_motion_data for d in motion]

    ax.plot(all_times, rms_torques, linewidth=2, color="darkred", label="RMS Torque")
    ax.axhline(SAFE_LIMIT_NM, color="red", linestyle="--", linewidth=2)

    ax2 = ax.twinx()
    ax2.plot(
        all_times,
        np.rad2deg(rms_errors),
        linewidth=2,
        color="blue",
        label="RMS Error",
    )

    for boundary_time, name in transition_boundaries:
        ax.axvline(boundary_time, color="gray", linestyle=":", alpha=0.5)

    add_pose_markers(ax)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS Torque (N·m)", color="darkred")
    ax2.set_ylabel("RMS Position Error (°)", color="blue")
    ax.set_title("RMS Torque and Position Error")
    ax.tick_params(axis="y", labelcolor="darkred")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax.grid(alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def run_motion_sequence(backend, kinematics, payload_g, output_dir):
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
            if joint_positions is None or np.any(np.isnan(joint_positions)):
                print(f"   ✗ {pose_dict['name']}: IK failed")
                return False
            joint_positions_sequence.append(joint_positions)
            print(f"   ✓ {pose_dict['name']}")
        except Exception as e:
            print(f"   ✗ {pose_dict['name']}: {e}")
            return False

    # Move to initial pose
    print("\n2. Moving to initial neutral pose...")
    mujoco.mj_resetData(backend.model, backend.data)
    initial_joints = joint_positions_sequence[0]
    backend.data.ctrl[:7] = initial_joints

    # Stabilize at initial pose
    for _ in range(1000):
        backend.data.ctrl[:7] = initial_joints
        mujoco.mj_step(backend.model, backend.data)

    print("   ✓ Reached initial pose")

    # Create renders directory
    renders_dir = output_dir / f"renders_{payload_g}g"
    renders_dir.mkdir(exist_ok=True)

    # Render initial pose
    initial_render_path = renders_dir / f"pose_0_{TEST_POSE_SEQUENCE[0]['key']}.png"
    render_robot_pose(backend, initial_render_path)
    print(f"   ✓ Saved initial pose render: {initial_render_path.name}")

    # Run through motion sequence
    print("\n3. Running motion sequence...")
    print("=" * 100)

    all_motion_data = []
    transition_names = []
    pose_achievements = [
        (0.0, TEST_POSE_SEQUENCE[0]["name"])
    ]  # Track when poses are achieved
    cumulative_time = 0.0

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

        # Prepare render path for achieved pose
        render_path = renders_dir / f"pose_{i+1}_{next_pose['key']}.png"

        # Simulate motion with tracking
        motion_data, achieved_time = simulate_motion_with_tracking(
            backend,
            current_joints,
            next_joints,
            MOTION_DURATION,
            transition_name,
            render_path,
        )

        all_motion_data.append(motion_data)

        # Track pose achievement
        cumulative_time += achieved_time
        pose_achievements.append((cumulative_time, next_pose["name"]))

        # Check for safety violations
        if motion_data:
            max_torque = max(d["max_torque"] for d in motion_data)
            if max_torque >= SAFE_LIMIT_NM:
                print(
                    f"\n  ⚠ WARNING: Torque {max_torque:.3f} N·m exceeded safe limit!"
                )

    print("\n" + "=" * 100)
    print("  ✓ Motion sequence complete!")

    # Save data
    print("\n4. Saving data...")

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
        render_dir=renders_dir,
    )
    print(f"   ✓ Saved graphs: {graph_file.name}")

    # Summary statistics
    print("\n5. Summary statistics:")
    total_samples = sum(len(d) for d in all_motion_data)
    all_max_torques = [d["max_torque"] for motion in all_motion_data for d in motion]
    all_max_pos_errors = [
        d["max_pos_error"] for motion in all_motion_data for d in motion
    ]
    all_rms_pos_errors = [
        d["rms_pos_error"] for motion in all_motion_data for d in motion
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

    return True


def main():
    """Main routine."""
    print("=" * 100)
    print("TRACK TORQUES DURING MOTION - MUJOCO SIMULATION")
    print("=" * 100)

    # Create output directory
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"sim_motion_tracking_results/test_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}\n")

    # Configuration
    print("=" * 100)
    print("TEST CONFIGURATION")
    print("=" * 100)
    print(f"\nScene: {SCENE}")
    print(f"Pose sequence: {len(TEST_POSE_SEQUENCE)} poses")
    print(f"Transitions: {len(TEST_POSE_SEQUENCE) - 1}")
    print(f"Motion duration per transition: {MOTION_DURATION}s")
    print(f"Simulation frequency: {SIMULATION_FREQUENCY} Hz")
    print(f"Control update frequency: {CONTROL_FREQUENCY} Hz (matches real robot)")
    print(f"Sampling frequency: {SAMPLING_FREQUENCY} Hz")
    print(f"Payload masses to test: {PAYLOAD_MASSES_G}")

    print("\n" + "=" * 100)
    print("STARTING SIMULATION TESTS")
    print("=" * 100)

    try:
        for payload_g in PAYLOAD_MASSES_G:
            print(f"\n\n{'#' * 100}")
            print(f"PAYLOAD: {payload_g}g")
            print(f"{'#' * 100}")

            # Initialize backend
            backend = MujocoBackend(
                scene=SCENE,
                check_collision=False,
                kinematics_engine="AnalyticalKinematics",
                headless=True,
                use_audio=False,
            )

            # Modify payload mass
            payload_kg = payload_g / 1000.0
            original_mass = modify_payload_mass(backend.model, payload_kg)
            print(
                f"\nSet payload mass: {payload_kg:.3f} kg (original: {original_mass:.3f} kg)"
            )

            kinematics = backend.head_kinematics

            # Run motion sequence
            success = run_motion_sequence(backend, kinematics, payload_g, output_dir)

            if not success:
                print("\nStopping tests")
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback

        traceback.print_exc()

    # Summary
    print("\n" + "=" * 100)
    print("COMPLETE")
    print("=" * 100)
    print(f"Results: {output_dir}/")
    print("=" * 100)


if __name__ == "__main__":
    main()
