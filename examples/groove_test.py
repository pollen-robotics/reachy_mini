import mujoco
import mujoco.viewer
import time
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from stewart_little_control import PlacoIK
from stewart_little_control.mujoco_utils import (
    get_homogeneous_matrix_from_euler,
    get_joints,
)

# --- Configuration ---
ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent
MODEL_XML_PATH = str(ROOT_PATH / "descriptions" / "stewart_little_magnet" / "scene.xml")
PLACO_IK_DESCR_PATH = str(ROOT_PATH / "descriptions" / "stewart_little_magnet" / "") 
OUTPUT_DIR = Path("./experiment_results")
OUTPUT_DIR.mkdir(exist_ok=True)

SIMULATION_TIMESTEP = 0.002
CONTROL_DECIMATION = 10
CONTROL_FREQUENCY_HZ = 1 / (SIMULATION_TIMESTEP * CONTROL_DECIMATION)

EXPERIMENT_DURATION_S = 10.0
MOTION_FREQUENCY_HZ = 0.5
MOTION_AMPLITUDE_POS = 0.05
MOTION_AMPLITUDE_ORI = np.deg2rad(30)

NEUTRAL_POSITION = np.array([0.0, 0.0, 0.155])
NEUTRAL_EULER_ANGLES = np.array([0.0, 0.0, 0.0]) # roll, pitch, yaw (radians)

ACTUATED_JOINT_NAMES = ["1", "2", "3", "4", "5", "6"]


# --- Plotting and Metrics ---
def plot_joint_tracking(time_vec, commanded_q, actual_q, dof_name, joint_names_plot, output_path):
    num_joints = commanded_q.shape[1]
    fig, axs = plt.subplots(num_joints, 1, figsize=(12, 2 * num_joints), sharex=True)
    if num_joints == 1:
        axs = [axs]

    for i in range(num_joints):
        axs[i].plot(time_vec, commanded_q[:, i], label=f'Commanded {joint_names_plot[i]}')
        axs[i].plot(time_vec, actual_q[:, i], label=f'Actual {joint_names_plot[i]}', linestyle='--')
        axs[i].set_ylabel('Joint Angle (rad)')
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Joint Tracking Performance for {dof_name} Oscillation', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot to {output_path}")

def calculate_tracking_error(commanded_q, actual_q):
    if commanded_q.shape[0] != actual_q.shape[0]:
        min_len = min(commanded_q.shape[0], actual_q.shape[0])
        commanded_q = commanded_q[:min_len]
        actual_q = actual_q[:min_len]
        print(f"Warning: Commanded and actual histories have different lengths. Truncated to {min_len}.")

    error = commanded_q - actual_q
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(error**2, axis=0))
    return mae, rmse

# --- Main Experiment Logic ---
def run_oscillation_experiment(model, data, viewer, placo_ik,
                               dof_to_test, base_pos, base_ori,
                               amplitude_pos, amplitude_ori, freq_hz, duration_s):
    print(f"\n--- Running experiment for DoF: {dof_to_test} ---")

    commanded_joint_history = []
    actual_joint_history = []
    time_history = []

    mujoco.mj_resetData(model, data)
    
    sim_steps = 0
    control_steps = 0
    experiment_start_wall_time = time.time()
    current_motion_time = 0.0

    while current_motion_time < duration_s:
        loop_start_time = time.time()
        current_motion_time = time.time() - experiment_start_wall_time

        if sim_steps % CONTROL_DECIMATION == 0:
            current_pose_params = {
                "position": base_pos.copy(),
                "euler_angles": base_ori.copy(),
            }
            
            oscillation_value = np.sin(2 * np.pi * freq_hz * current_motion_time)

            if dof_to_test == "X":
                current_pose_params["position"][0] += amplitude_pos * oscillation_value
            elif dof_to_test == "Y":
                current_pose_params["position"][1] += amplitude_pos * oscillation_value
            elif dof_to_test == "Z":
                current_pose_params["position"][2] += amplitude_pos * oscillation_value
            elif dof_to_test == "Roll":
                current_pose_params["euler_angles"][0] += amplitude_ori * oscillation_value
            elif dof_to_test == "Pitch":
                current_pose_params["euler_angles"][1] += amplitude_ori * oscillation_value
            elif dof_to_test == "Yaw":
                current_pose_params["euler_angles"][2] += amplitude_ori * oscillation_value

            target_pose_matrix = get_homogeneous_matrix_from_euler(
                position=tuple(current_pose_params["position"]),
                euler_angles=tuple(current_pose_params["euler_angles"]),
            )

            try:
                target_angles_rad = placo_ik.ik(target_pose_matrix)
            except Exception as e:
                print(f"IK failed at t={current_motion_time:.2f}s for {dof_to_test}: {e}")
                if commanded_joint_history:
                    target_angles_rad = commanded_joint_history[-1]
                else:
                    print("IK failed on first attempt, skipping control command.")
                    mujoco.mj_step(model, data)
                    sim_steps += 1
                    if viewer and viewer.is_running(): viewer.sync()
                    time.sleep(max(0, SIMULATION_TIMESTEP - (time.time() - loop_start_time)))
                    continue
            
            data.ctrl[:] = target_angles_rad

            commanded_joint_history.append(target_angles_rad.copy())
            actual_q = get_joints(model, data) # Using your provided function
            actual_joint_history.append(actual_q.copy())
            time_history.append(current_motion_time)
            control_steps +=1

        mujoco.mj_step(model, data)
        sim_steps += 1

        if viewer and viewer.is_running():
            viewer.sync()

        elapsed_loop_time = time.time() - loop_start_time
        time.sleep(max(0, SIMULATION_TIMESTEP - elapsed_loop_time))

    print(f"Experiment for {dof_to_test} completed. Ran {control_steps} control steps.")
    return (
        np.array(time_history),
        np.array(commanded_joint_history),
        np.array(actual_joint_history),
    )

# --- Main Script ---
if __name__ == "__main__":
    print("Initializing MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = SIMULATION_TIMESTEP

    print("Initializing PlacoIK...")
    try:
        placo_ik = PlacoIK(PLACO_IK_DESCR_PATH)
        if len(ACTUATED_JOINT_NAMES) != 6:
             raise ValueError("ACTUATED_JOINT_NAMES must contain 6 joint names.")
    except Exception as e:
        print(f"Error initializing PlacoIK: {e}")
        exit()

    neutral_pose_matrix = get_homogeneous_matrix_from_euler(
        position=tuple(NEUTRAL_POSITION),
        euler_angles=tuple(NEUTRAL_EULER_ANGLES),
    )
    try:
        initial_angles_rad = placo_ik.ik(neutral_pose_matrix)
        data.ctrl[:] = initial_angles_rad
        for _ in range(int(0.1 / SIMULATION_TIMESTEP)):
            mujoco.mj_step(model, data)
    except Exception as e:
        print(f"IK failed for initial neutral pose: {e}. Starting with zero controls.")
        data.ctrl[:] = 0.0

    print(f"Control frequency: {CONTROL_FREQUENCY_HZ:.2f} Hz")
    print(f"Actuated joint names being used for plotting/metrics: {ACTUATED_JOINT_NAMES}")

    viewer = None
    try:
        print("Launching MuJoCo viewer...")
        viewer = mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        )
        time.sleep(1)
    except Exception as e:
        print(f"Could not launch viewer: {e}. Running headless.")


    dofs_to_test = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    all_metrics = {}

    for dof in dofs_to_test:
        if viewer and not viewer.is_running():
            print("Viewer closed, exiting.")
            break

        time_vec, commanded_q, actual_q = run_oscillation_experiment(
            model, data, viewer, placo_ik,
            dof, NEUTRAL_POSITION, NEUTRAL_EULER_ANGLES,
            MOTION_AMPLITUDE_POS, MOTION_AMPLITUDE_ORI,
            MOTION_FREQUENCY_HZ, EXPERIMENT_DURATION_S
        )

        if time_vec.size == 0:
            print(f"Skipping metrics and plotting for {dof} due to empty data.")
            continue

        mae, rmse = calculate_tracking_error(commanded_q, actual_q)
        all_metrics[dof] = {"MAE": mae, "RMSE": rmse}

        print(f"\nMetrics for {dof}:")
        for i, name in enumerate(ACTUATED_JOINT_NAMES):
            print(f"  Joint {name}: MAE = {mae[i]:.4f} rad, RMSE = {rmse[i]:.4f} rad")
        print(f"  Overall Avg MAE: {np.mean(mae):.4f} rad, Overall Avg RMSE: {np.mean(rmse):.4f} rad")

        plot_file_name = OUTPUT_DIR / f"tracking_{dof.lower()}_oscillation.png"
        plot_joint_tracking(time_vec, commanded_q, actual_q, dof, ACTUATED_JOINT_NAMES, plot_file_name)
        
        if viewer: time.sleep(0.5) # Brief pause if viewer is active


    print("\n--- Experiment Summary ---")
    for dof, metrics in all_metrics.items():
        print(f"DoF: {dof}")
        print(f"  Avg MAE across joints: {np.mean(metrics['MAE']):.4f} rad")
        print(f"  Avg RMSE across joints: {np.mean(metrics['RMSE']):.4f} rad")

    if viewer and viewer.is_running():
        print("\nExperiments finished. Close the viewer to exit.")
        while viewer.is_running():
            time.sleep(0.1)
    else:
        print("\nExperiments finished.")

    if viewer:
        viewer.close()