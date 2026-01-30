#!/usr/bin/env python3
"""
Find Maximum Payload for Each Pose

For each pose, find the maximum payload mass (inside head or on top)
before reaching the safe torque limit (0.45 N·m).

Uses binary search for efficiency.
"""

import sys
import numpy as np
import mujoco
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from reachy_mini.daemon.backend.mujoco.backend import MujocoBackend
from reachy_mini.utils import create_head_pose

# ============================================================================
# CONFIGURATION
# ============================================================================

# Test poses
TEST_POSES = {
    "neutral": {
        "name": "Neutral (0°, 0°)",
        "pose": create_head_pose(roll=0, pitch=0, yaw=0, degrees=True),
    },
    "pitch_up_30": {
        "name": "Pitch up 30° (head looking up)",
        "pose": create_head_pose(roll=0, pitch=-30, yaw=0, degrees=True),
    },
    "pitch_down_30": {
        "name": "Pitch down 30° (head looking down)",
        "pose": create_head_pose(roll=0, pitch=30, yaw=0, degrees=True),
    },
    "roll_right_30": {
        "name": "Roll right 30°",
        "pose": create_head_pose(roll=30, pitch=0, yaw=0, degrees=True),
    },
    "roll_left_30": {
        "name": "Roll left 30°",
        "pose": create_head_pose(roll=-30, pitch=0, yaw=0, degrees=True),
    },
    "pitch_up_15": {
        "name": "Pitch up 15°",
        "pose": create_head_pose(roll=0, pitch=-15, yaw=0, degrees=True),
    },
    "pitch_down_15": {
        "name": "Pitch down 15°",
        "pose": create_head_pose(roll=0, pitch=15, yaw=0, degrees=True),
    },
    "roll_right_15": {
        "name": "Roll right 15°",
        "pose": create_head_pose(roll=15, pitch=0, yaw=0, degrees=True),
    },
    "roll_left_15": {
        "name": "Roll left 15°",
        "pose": create_head_pose(roll=-15, pitch=0, yaw=0, degrees=True),
    },
    "translate_up_2cm": {
        "name": "Translate up 2cm",
        "pose": create_head_pose(
            x=0, y=0, z=0.02, roll=0, pitch=0, yaw=0, degrees=True
        ),
    },
    "translate_down_1cm": {
        "name": "Translate down 1cm",
        "pose": create_head_pose(
            x=0, y=0, z=-0.01, roll=0, pitch=0, yaw=0, degrees=True
        ),
    },
    "translate_forward_1cm": {
        "name": "Translate forward 1cm",
        "pose": create_head_pose(
            x=0.01, y=0, z=0, roll=0, pitch=0, yaw=0, degrees=True
        ),
    },
    "translate_side_1cm": {
        "name": "Translate side 1cm",
        "pose": create_head_pose(
            x=0, y=0.01, z=0, roll=0, pitch=0, yaw=0, degrees=True
        ),
    },
}

# Torque limits
STALL_TORQUE_NM = 0.6
SAFE_LIMIT_NM = 0.45  # 75% of stall

# Binary search parameters
MIN_MASS_KG = 0.0
MAX_MASS_KG = 10.0
MASS_TOLERANCE_G = 5  # Stop when range is within 5g

MAX_STEPS = 5000
CONVERGENCE_POS_TOL = 1e-6
CONVERGENCE_VEL_TOL = 1e-5

# ============================================================================
# MASS TESTING
# ============================================================================


def test_pose_with_mass(backend, kinematics, pose_dict, mass_kg, body_name):
    """
    Test a pose with a specific mass and return max torque.

    Args:
        backend: MujocoBackend instance
        kinematics: Head kinematics
        pose_dict: Pose dictionary with 'pose' key
        mass_kg: Mass to test in kg
        body_name: Body name to modify ("xl_330" or "fake_payload_head")

    Returns:
        max_torque (float) or None if IK failed
    """
    pose = pose_dict["pose"]

    # Compute IK
    joint_positions = kinematics.ik(pose, body_yaw=0.0)
    if joint_positions is None or np.any(np.isnan(joint_positions)):
        return None

    # Reset simulation
    mujoco.mj_resetData(backend.model, backend.data)
    backend.data.ctrl[:7] = joint_positions

    # Stabilize
    for _ in range(500):
        backend.data.ctrl[:7] = joint_positions
        mujoco.mj_step(backend.model, backend.data)

    # Set mass
    body_id = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    backend.model.body_mass[body_id] = mass_kg

    # Converge
    prev_qpos = backend.data.qpos[backend.joint_qpos_addr[:7]].copy()
    for step in range(MAX_STEPS):
        backend.data.ctrl[:7] = joint_positions
        mujoco.mj_step(backend.model, backend.data)

        if step % 50 == 0 and step > 0:
            curr_qpos = backend.data.qpos[backend.joint_qpos_addr[:7]].copy()
            curr_qvel = backend.data.qvel[backend.joint_qpos_addr[:7]].copy()

            pos_change = np.max(np.abs(curr_qpos - prev_qpos))
            max_vel = np.max(np.abs(curr_qvel))

            if pos_change < CONVERGENCE_POS_TOL and max_vel < CONVERGENCE_VEL_TOL:
                break

            prev_qpos = curr_qpos

    # Get max torque
    torques = []
    for i in range(7):
        torques.append(backend.data.actuator_force[i])

    max_torque = float(np.max(np.abs(torques)))
    return max_torque


def find_max_mass_for_pose(pose_key, pose_dict, scene, body_name):
    """
    Use binary search to find maximum mass for a pose.

    Args:
        pose_key: Pose identifier
        pose_dict: Pose configuration
        scene: Scene name ("empty" or "empty_payload")
        body_name: Body to modify mass

    Returns:
        dict with results
    """
    print(f"\n{'='*60}")
    print(f"Testing: {pose_dict['name']}")
    print(f"{'='*60}")

    # Initialize backend
    backend = MujocoBackend(
        scene=scene,
        check_collision=False,
        kinematics_engine="AnalyticalKinematics",
        headless=True,
        use_audio=False,
    )
    kinematics = backend.head_kinematics

    # Binary search
    min_mass = MIN_MASS_KG
    max_mass = MAX_MASS_KG
    iterations = 0

    while (max_mass - min_mass) * 1000 > MASS_TOLERANCE_G:
        mid_mass = (min_mass + max_mass) / 2
        iterations += 1

        max_torque = test_pose_with_mass(
            backend, kinematics, pose_dict, mid_mass, body_name
        )

        if max_torque is None:
            print(f"  Iteration {iterations}: {mid_mass*1000:.1f}g - IK FAILED")
            max_mass = mid_mass
            continue

        pct_stall = (max_torque / STALL_TORQUE_NM) * 100
        status = "✓ SAFE" if max_torque < SAFE_LIMIT_NM else "✗ LIMIT"

        print(
            f"  Iteration {iterations}: {mid_mass*1000:.1f}g -> {max_torque:.4f} N·m ({pct_stall:.1f}% stall) {status}"
        )

        if max_torque < SAFE_LIMIT_NM:
            # Safe, try higher
            min_mass = mid_mass
        else:
            # Over limit, try lower
            max_mass = mid_mass

    # Final mass is the safe one
    final_mass_kg = min_mass
    final_torque = test_pose_with_mass(
        backend, kinematics, pose_dict, final_mass_kg, body_name
    )

    result = {
        "pose_key": pose_key,
        "pose_name": pose_dict["name"],
        "max_mass_kg": final_mass_kg,
        "max_mass_g": final_mass_kg * 1000,
        "max_torque_nm": final_torque,
        "pct_of_stall": (final_torque / STALL_TORQUE_NM) * 100,
        "iterations": iterations,
    }

    print(f"\n  ✓ Maximum mass: {result['max_mass_g']:.1f}g")
    print(
        f"    Torque at limit: {result['max_torque_nm']:.4f} N·m ({result['pct_of_stall']:.1f}% stall)"
    )

    return result


# ============================================================================
# VISUALIZATION
# ============================================================================


def create_comparison_figure(results_inside, results_ontop, output_path):
    """Create comparison figure of max payload for each pose."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Sort by max mass
    results_inside_sorted = sorted(results_inside, key=lambda r: r["max_mass_g"])
    results_ontop_sorted = sorted(results_ontop, key=lambda r: r["max_mass_g"])

    # Plot 1: Inside head
    pose_names_inside = [r["pose_name"] for r in results_inside_sorted]
    masses_inside = [r["max_mass_g"] for r in results_inside_sorted]

    bars1 = ax1.barh(
        range(len(pose_names_inside)), masses_inside, color="steelblue", alpha=0.8
    )
    ax1.set_yticks(range(len(pose_names_inside)))
    ax1.set_yticklabels(pose_names_inside, fontsize=10)
    ax1.set_xlabel("Maximum Payload Mass (g)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Maximum Payload Inside Head\n(modifying xl_330 mass)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, masses_inside)):
        ax1.text(
            val + 20,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}g",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # Plot 2: On top of head
    pose_names_ontop = [r["pose_name"] for r in results_ontop_sorted]
    masses_ontop = [r["max_mass_g"] for r in results_ontop_sorted]

    bars2 = ax2.barh(
        range(len(pose_names_ontop)), masses_ontop, color="coral", alpha=0.8
    )
    ax2.set_yticks(range(len(pose_names_ontop)))
    ax2.set_yticklabels(pose_names_ontop, fontsize=10)
    ax2.set_xlabel("Maximum Payload Mass (g)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Maximum Payload On Top of Head\n(fake_payload_head mass)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, masses_ontop)):
        ax2.text(
            val + 20,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}g",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Find max payload per pose")
    parser.add_argument(
        "--mode",
        choices=["inside", "ontop", "both"],
        default="both",
        help="Test mode: inside head, on top, or both",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("MAXIMUM PAYLOAD PER POSE")
    print("=" * 80)
    print(
        f"\nSafe torque limit: {SAFE_LIMIT_NM} N·m ({(SAFE_LIMIT_NM/STALL_TORQUE_NM)*100:.0f}% of stall)"
    )
    print(f"Test poses: {len(TEST_POSES)}")
    print(f"Binary search tolerance: {MASS_TOLERANCE_G}g\n")

    results_inside = []
    results_ontop = []

    # Test inside head
    if args.mode in ["inside", "both"]:
        print("\n" + "=" * 80)
        print("TESTING: PAYLOAD INSIDE HEAD (xl_330 mass)")
        print("=" * 80)

        for pose_key, pose_dict in TEST_POSES.items():
            result = find_max_mass_for_pose(pose_key, pose_dict, "empty", "xl_330")
            results_inside.append(result)

    # Test on top of head
    if args.mode in ["ontop", "both"]:
        print("\n" + "=" * 80)
        print("TESTING: PAYLOAD ON TOP OF HEAD (fake_payload_head mass)")
        print("=" * 80)

        for pose_key, pose_dict in TEST_POSES.items():
            result = find_max_mass_for_pose(
                pose_key, pose_dict, "empty_payload", "fake_payload_head"
            )
            results_ontop.append(result)

    # Save results
    output_dir = Path(".")

    if results_inside:
        np.save(
            output_dir / "max_payload_inside.npy", results_inside, allow_pickle=True
        )
        print(f"\n✓ Inside head results saved: max_payload_inside.npy")

    if results_ontop:
        np.save(output_dir / "max_payload_ontop.npy", results_ontop, allow_pickle=True)
        print(f"✓ On top results saved: max_payload_ontop.npy")

    # Create comparison figure
    if results_inside and results_ontop:
        print(f"\nGenerating comparison figure...")
        create_comparison_figure(
            results_inside, results_ontop, output_dir / "max_payload_comparison.png"
        )
        print(f"✓ Comparison figure saved: max_payload_comparison.png")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results_inside:
        print("\nINSIDE HEAD:")
        min_result = min(results_inside, key=lambda r: r["max_mass_g"])
        max_result = max(results_inside, key=lambda r: r["max_mass_g"])
        avg_mass = np.mean([r["max_mass_g"] for r in results_inside])

        print(
            f"  Most restrictive: {min_result['pose_name']} - {min_result['max_mass_g']:.0f}g"
        )
        print(
            f"  Least restrictive: {max_result['pose_name']} - {max_result['max_mass_g']:.0f}g"
        )
        print(f"  Average: {avg_mass:.0f}g")

    if results_ontop:
        print("\nON TOP OF HEAD:")
        min_result = min(results_ontop, key=lambda r: r["max_mass_g"])
        max_result = max(results_ontop, key=lambda r: r["max_mass_g"])
        avg_mass = np.mean([r["max_mass_g"] for r in results_ontop])

        print(
            f"  Most restrictive: {min_result['pose_name']} - {min_result['max_mass_g']:.0f}g"
        )
        print(
            f"  Least restrictive: {max_result['pose_name']} - {max_result['max_mass_g']:.0f}g"
        )
        print(f"  Average: {avg_mass:.0f}g")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
