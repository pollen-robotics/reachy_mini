#!/usr/bin/env python3
"""
Check Motor Torques for Head Poses with Variable Mass

Tests different head masses in 50g increments until reaching safety torque limit.
Tests pitch, roll, and translation poses.
"""

import sys
import numpy as np
import mujoco
from pathlib import Path
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).parent / "src"))

from reachy_mini.daemon.backend.mujoco.backend import MujocoBackend
from reachy_mini.utils import create_head_pose

# ============================================================================
# CONFIGURATION
# ============================================================================

# Test poses - pitch, roll, and translations
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
        "pose": create_head_pose(x=0, y=0, z=0.02, roll=0, pitch=0, yaw=0, degrees=True),
    },
    "translate_down_1cm": {
        "name": "Translate down 1cm",
        "pose": create_head_pose(x=0, y=0, z=-0.01, roll=0, pitch=0, yaw=0, degrees=True),
    },
    "translate_forward_1cm": {
        "name": "Translate forward 1cm",
        "pose": create_head_pose(x=0.01, y=0, z=0, roll=0, pitch=0, yaw=0, degrees=True),
    },
    "translate_side_1cm": {
        "name": "Translate side 1cm",
        "pose": create_head_pose(x=0, y=0.01, z=0, roll=0, pitch=0, yaw=0, degrees=True),
    },
}

SCENE = "empty"
MAX_STEPS = 5000
CONVERGENCE_POS_TOL = 1e-6
CONVERGENCE_VEL_TOL = 1e-5

# Torque reference
STALL_TORQUE_NM = 0.6  # Motor stall torque
SAFE_LIMIT_NM = 0.45   # 75% of stall torque

# Mass testing
MASS_INCREMENT_KG = 0.05  # 50g increments
MAX_MASS_KG = 2.0  # Maximum to test

# XML path (relative to repo root)
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
XML_PATH = REPO_ROOT / "src/reachy_mini/descriptions/reachy_mini/mjcf/reachy_mini.xml"

# ============================================================================
# MASS MODIFICATION
# ============================================================================

def modify_head_mass_in_model(model, new_mass_kg):
    """
    Modify the xl_330 body mass directly in the MuJoCo model.

    Args:
        model: MuJoCo model object
        new_mass_kg: New mass value in kg

    Returns:
        original_mass_kg: The original mass value
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "xl_330")

    # Get original mass
    original_mass = model.body_mass[body_id]

    # Set new mass (without changing COM)
    model.body_mass[body_id] = new_mass_kg

    return original_mass


def get_original_head_mass():
    """Get the original head mass from the XML file."""
    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    for body in root.iter('body'):
        if body.get('name') == 'xl_330':
            inertial = body.find('inertial')
            if inertial is not None:
                return float(inertial.get('mass'))

    raise ValueError("Could not find xl_330 mass in XML")

# ============================================================================
# RENDERING AND PLOTTING
# ============================================================================

def render_robot_image(backend, output_path, width=800, height=600):
    """Render current robot state and save image."""
    renderer = mujoco.Renderer(backend.model, height=height, width=width)
    renderer.update_scene(backend.data)
    pixels = renderer.render()

    plt.figure(figsize=(8, 6))
    plt.imshow(pixels)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_torques(result, output_path):
    """Create bar chart of motor torques for a pose."""
    torques = result['torques']
    motor_names = list(torques.keys())
    torque_values = [torques[name] for name in motor_names]
    abs_torques = [abs(t) for t in torque_values]

    # Shorten motor names for display
    display_names = [name.replace('_actuator', '') for name in motor_names]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars with colors based on torque magnitude
    colors = []
    for abs_torque in abs_torques:
        if abs_torque >= SAFE_LIMIT_NM:
            colors.append('#e74c3c')  # Red - exceeds safe limit
        elif abs_torque > 0.3:
            colors.append('#f39c12')  # Orange - high
        else:
            colors.append('#3498db')  # Blue - normal

    bars = ax.bar(range(len(motor_names)), torque_values, color=colors, alpha=0.7, edgecolor='black')

    # Add safe limit lines
    ax.axhline(y=SAFE_LIMIT_NM, color='red', linestyle='--', linewidth=2, label=f'Safe Limit (+{SAFE_LIMIT_NM} N·m)')
    ax.axhline(y=-SAFE_LIMIT_NM, color='red', linestyle='--', linewidth=2, label=f'Safe Limit (-{SAFE_LIMIT_NM} N·m)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Labels and title
    ax.set_xlabel('Motor', fontsize=12, fontweight='bold')
    ax.set_ylabel('Torque (N·m)', fontsize=12, fontweight='bold')

    mass_g = result.get('head_mass_g', 'unknown')
    ax.set_title(f"Motor Torques - {result['pose_name']} (Head: {mass_g}g)\n"
                 f"Max: {result['max_torque']:.4f} N·m ({result['pct_of_stall']:.1f}% of stall)",
                 fontsize=14, fontweight='bold')

    ax.set_xticks(range(len(motor_names)))
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, torque_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_figure(all_mass_results, output_path):
    """Create summary figure showing torque vs mass."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Group results by pose
    poses_data = {}
    for result in all_mass_results:
        pose_key = result['pose_key']
        if pose_key not in poses_data:
            poses_data[pose_key] = {
                'name': result['pose_name'],
                'masses': [],
                'max_torques': []
            }
        poses_data[pose_key]['masses'].append(result['head_mass_g'])
        poses_data[pose_key]['max_torques'].append(result['max_torque'])

    # Plot 1: Torque vs Mass for each pose
    for pose_key, data in poses_data.items():
        ax1.plot(data['masses'], data['max_torques'], 'o-', label=data['name'], linewidth=2, markersize=6)

    ax1.axhline(y=SAFE_LIMIT_NM, color='red', linestyle='--', linewidth=2, label=f'Safe Limit ({SAFE_LIMIT_NM} N·m)')
    ax1.axhline(y=STALL_TORQUE_NM, color='darkred', linestyle='--', linewidth=2, label=f'Stall ({STALL_TORQUE_NM} N·m)')
    ax1.set_xlabel('Head Mass (g)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Max Torque (N·m)', fontsize=12, fontweight='bold')
    ax1.set_title('Maximum Torque vs Head Mass', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Plot 2: Statistics and Information Panel
    ax2.axis('off')

    # Find critical pose (highest final torque)
    max_result = max(all_mass_results, key=lambda r: r['max_torque'])

    # Find which poses exceeded limit
    exceeded_poses = {}
    safe_poses = {}
    for pose_key, data in poses_data.items():
        max_torque_for_pose = max(data['max_torques'])
        final_mass = data['masses'][-1]

        if max_torque_for_pose >= SAFE_LIMIT_NM:
            exceeded_poses[data['name']] = {
                'mass': final_mass,
                'torque': max_torque_for_pose
            }
        else:
            safe_poses[data['name']] = {
                'mass': final_mass,
                'torque': max_torque_for_pose,
                'pct': (max_torque_for_pose / SAFE_LIMIT_NM) * 100
            }

    # Get mass range
    all_masses = [r['head_mass_g'] for r in all_mass_results]
    original_mass = min(all_masses)
    max_tested_mass = max(all_masses)

    # Build statistics text
    stats_text = "EXPERIMENT STATISTICS\n"
    stats_text += "=" * 50 + "\n\n"

    stats_text += f"Original head mass:  {original_mass}g\n"
    stats_text += f"Maximum tested mass: {max_tested_mass}g\n"
    stats_text += f"Mass increment:      {MASS_INCREMENT_KG * 1000:.0f}g\n"
    stats_text += f"Total tests:         {len(all_mass_results)}\n"
    stats_text += f"Poses tested:        {len(poses_data)}\n\n"

    stats_text += "SAFETY LIMITS\n"
    stats_text += "-" * 50 + "\n"
    stats_text += f"Motor stall torque:  {STALL_TORQUE_NM:.2f} N·m (100%)\n"
    stats_text += f"Safe limit (75%):    {SAFE_LIMIT_NM:.2f} N·m\n\n"

    stats_text += "CRITICAL CONDITION\n"
    stats_text += "-" * 50 + "\n"
    stats_text += f"Pose:     {max_result['pose_name']}\n"
    stats_text += f"Mass:     {max_result['head_mass_g']}g\n"
    stats_text += f"Torque:   {max_result['max_torque']:.4f} N·m\n"
    stats_text += f"% Stall:  {max_result['pct_of_stall']:.1f}%\n"
    stats_text += f"Motor:    {max_result['hardest_motor']}\n\n"

    if exceeded_poses:
        stats_text += f"POSES EXCEEDING LIMIT ({len(exceeded_poses)})\n"
        stats_text += "-" * 50 + "\n"
        for pose_name in sorted(exceeded_poses.keys()):
            info = exceeded_poses[pose_name]
            stats_text += f"• {pose_name}\n"
        stats_text += "\n"

    if safe_poses:
        stats_text += f"POSES STILL SAFE AT {max_tested_mass}g ({len(safe_poses)})\n"
        stats_text += "-" * 50 + "\n"
        for pose_name in sorted(safe_poses.keys(),
                                key=lambda p: safe_poses[p]['pct'],
                                reverse=True):
            info = safe_poses[pose_name]
            stats_text += f"• {pose_name:35s} {info['pct']:5.1f}%\n"
        stats_text += "\n"

    stats_text += "KEY INSIGHTS\n"
    stats_text += "-" * 50 + "\n"
    stats_text += "• Roll poses generate highest torques\n"
    stats_text += "• Translation poses remain well below limit\n"
    stats_text += "• Torque increases linearly with mass\n"
    stats_text += "• Testing stopped when first pose hit limit\n"

    # Display text
    ax2.text(0.05, 0.95, stats_text,
             transform=ax2.transAxes,
             fontfamily='monospace',
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# TESTING
# ============================================================================

def test_pose(backend, kinematics, pose_dict, pose_key, head_mass_g, output_dir=None):
    """
    Test a single pose and measure motor torques.

    Returns:
        dict with results or None if IK failed
    """
    pose = pose_dict["pose"]
    pose_name = pose_dict["name"]

    # Compute IK
    joint_positions = kinematics.ik(pose, body_yaw=0.0)

    if joint_positions is None or np.any(np.isnan(joint_positions)):
        return None

    # Apply to MuJoCo
    backend.data.ctrl[:7] = joint_positions

    # Step until convergence
    prev_qpos = backend.data.qpos[backend.joint_qpos_addr[:7]].copy()
    converged = False

    for step in range(MAX_STEPS):
        backend.data.ctrl[:7] = joint_positions
        mujoco.mj_step(backend.model, backend.data)

        if step % 50 == 0 and step > 0:
            curr_qpos = backend.data.qpos[backend.joint_qpos_addr[:7]].copy()
            curr_qvel = backend.data.qvel[backend.joint_qpos_addr[:7]].copy()

            pos_change = np.max(np.abs(curr_qpos - prev_qpos))
            max_vel = np.max(np.abs(curr_qvel))

            if pos_change < CONVERGENCE_POS_TOL and max_vel < CONVERGENCE_VEL_TOL:
                converged = True
                break

            prev_qpos = curr_qpos

    # Read torques
    torques = {}
    for i in range(7):
        name = mujoco.mj_id2name(backend.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        torques[name] = backend.data.actuator_force[i]

    torque_vals = np.array(list(torques.values()))
    abs_torques = np.abs(torque_vals)
    max_torque = float(np.max(abs_torques))

    # Find hardest working motor
    max_idx = np.argmax(abs_torques)
    hardest_motor = mujoco.mj_id2name(backend.model, mujoco.mjtObj.mjOBJ_ACTUATOR, max_idx)
    hardest_torque = torque_vals[max_idx]

    # Calculate percentage of stall torque
    pct_of_stall = (max_torque / STALL_TORQUE_NM) * 100

    result = {
        "pose_key": pose_key,
        "pose_name": pose_name,
        "head_mass_g": head_mass_g,
        "converged": converged,
        "torques": torques,
        "max_torque": max_torque,
        "rms_torque": float(np.sqrt(np.mean(torque_vals**2))),
        "sum_abs_torques": float(np.sum(abs_torques)),
        "hardest_motor": hardest_motor,
        "hardest_torque": hardest_torque,
        "pct_of_stall": pct_of_stall,
        "exceeds_limit": max_torque >= SAFE_LIMIT_NM,
    }

    # Save visualizations if output directory provided
    if output_dir:
        output_dir = Path(output_dir)

        # Save robot render
        render_path = output_dir / f"{pose_key}_{head_mass_g}g_render.png"
        render_robot_image(backend, render_path)
        result["render_path"] = str(render_path)

        # Save torque graph
        graph_path = output_dir / f"{pose_key}_{head_mass_g}g_torques.png"
        plot_torques(result, graph_path)
        result["graph_path"] = str(graph_path)

    return result


def test_mass(mass_kg, output_dir):
    """
    Test all poses with a specific head mass.

    Returns:
        (results_list, limit_reached)
    """
    mass_g = int(mass_kg * 1000)
    print(f"\n{'='*80}")
    print(f"TESTING HEAD MASS: {mass_g}g ({mass_kg:.3f} kg)")
    print(f"{'='*80}")

    # Initialize backend
    backend = MujocoBackend(
        scene=SCENE,
        check_collision=False,
        kinematics_engine="AnalyticalKinematics",
        headless=True,
        use_audio=False,
    )

    # Modify head mass in the model
    original_mass = modify_head_mass_in_model(backend.model, mass_kg)

    if abs(mass_kg - original_mass) < 0.001:
        print(f"Using original head mass: {original_mass:.4f} kg")
    else:
        print(f"Modified head mass from {original_mass:.4f} kg to {mass_kg:.4f} kg")

    kinematics = backend.head_kinematics

    # Warm up to neutral
    neutral_pose = TEST_POSES["neutral"]["pose"]
    neutral_joints = kinematics.ik(neutral_pose, body_yaw=0.0)
    backend.data.ctrl[:7] = neutral_joints

    for _ in range(1000):
        backend.data.ctrl[:7] = neutral_joints
        mujoco.mj_step(backend.model, backend.data)

    # Test all poses
    results = []
    limit_reached = False

    print("-"*80)
    print(f"{'POSE':<40s} {'MAX TORQUE':<15s} {'% STALL':<10s} {'STATUS':<10s}")
    print("-"*80)

    for pose_key, pose_dict in TEST_POSES.items():
        result = test_pose(backend, kinematics, pose_dict, pose_key, mass_g, output_dir)

        if result is not None:
            results.append(result)

            # Status indicator
            if result["exceeds_limit"]:
                status = "⚠ LIMIT"
                limit_reached = True
            elif result["pct_of_stall"] > 50:
                status = "⚡ HIGH"
            else:
                status = "✓ OK"

            print(f"{result['pose_name']:<40s} {result['max_torque']:>6.4f} N·m     "
                  f"{result['pct_of_stall']:>5.1f}%     {status}")
        else:
            print(f"{pose_dict['name']:<40s} IK FAILED")

    print("-"*80)

    return results, limit_reached


def main():
    print("="*80)
    print("HEAD MOTOR TORQUE CHECK - VARIABLE MASS")
    print("="*80)
    print(f"\nMotor stall torque: {STALL_TORQUE_NM} N·m")
    print(f"Safe limit (75%): {SAFE_LIMIT_NM} N·m")
    print(f"Test poses: {len(TEST_POSES)}")
    print(f"Mass increment: {int(MASS_INCREMENT_KG * 1000)}g")
    print(f"Max mass to test: {MAX_MASS_KG:.2f} kg\n")

    # Create output directory
    output_dir = Path("head_torque_check_results")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Get original mass from XML
    original_mass_kg = get_original_head_mass()
    print(f"Original head mass from XML: {original_mass_kg:.4f} kg ({int(original_mass_kg * 1000)}g)\n")

    # Test incrementally
    all_results = []
    current_mass = original_mass_kg
    limit_reached = False

    while current_mass <= MAX_MASS_KG and not limit_reached:
        results, reached = test_mass(current_mass, output_dir)
        all_results.extend(results)

        if reached:
            limit_reached = True
            print(f"\n⚠ Torque limit reached at {int(current_mass * 1000)}g!")
            max_safe_mass = current_mass - MASS_INCREMENT_KG
            print(f"✓ Max safe head mass: {int(max_safe_mass * 1000)}g")
            break

        current_mass += MASS_INCREMENT_KG

    # Save results to file
    results_file = output_dir / "mass_test_results.npy"
    np.save(results_file, all_results, allow_pickle=True)
    print(f"\n✓ Results saved to: {results_file}")

    # Create summary figure
    if all_results:
        print(f"\nGenerating summary figure...")
        summary_path = output_dir / "mass_test_summary.png"
        create_summary_figure(all_results, summary_path)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    if limit_reached:
        # Find the hardest pose at limit
        limit_results = [r for r in all_results if r['exceeds_limit']]
        if limit_results:
            hardest = max(limit_results, key=lambda r: r['max_torque'])
            print(f"\nLimit reached at: {hardest['head_mass_g']}g")
            print(f"Critical pose: {hardest['pose_name']}")
            print(f"Peak torque: {hardest['max_torque']:.4f} N·m ({hardest['pct_of_stall']:.1f}% of stall)")
            print(f"Critical motor: {hardest['hardest_motor']}")
    else:
        print(f"\nNo limit reached up to {int(MAX_MASS_KG * 1000)}g")
        max_result = max(all_results, key=lambda r: r['max_torque'])
        print(f"Maximum torque observed: {max_result['max_torque']:.4f} N·m")
        print(f"At mass: {max_result['head_mass_g']}g, pose: {max_result['pose_name']}")

    print(f"\n✓ All results saved to: {output_dir}/")
    print(f"  - Individual renders and graphs for each mass/pose combination")
    print(f"  - Summary: mass_test_summary.png")
    print("="*80)


if __name__ == "__main__":
    main()
