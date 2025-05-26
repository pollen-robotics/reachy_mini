import mujoco
import mujoco.viewer
import time
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse # For CLI arguments

from stewart_little_control import PlacoIK
from stewart_little_control.mujoco_utils import (
    get_homogeneous_matrix_from_euler,
    get_joints,
)
from dance_moves import AVAILABLE_DANCE_MOVES, get_move_parameters

# --- Default Configuration (can be overridden by CLI args) ---
DEFAULT_TEST_DURATION_S = 10.0
DEFAULT_DANCE_DURATION_PER_MOVE_S = 15.0 # Used if --all_dances or as default for single dance
DEFAULT_DANCE_BPM = 124.0
DEFAULT_DANCE_GLOBAL_AMPLITUDE_SCALE = 1.0

# --- Static Configuration ---
ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent
MODEL_XML_PATH = str(ROOT_PATH / "descriptions" / "stewart_little_magnet" / "scene.xml")
PLACO_IK_DESCR_PATH = str(ROOT_PATH / "descriptions" / "stewart_little_magnet" / "")
OUTPUT_DIR = Path("./experiment_results") # For test_limits mode
OUTPUT_DIR.mkdir(exist_ok=True)

SIMULATION_TIMESTEP = 0.002
CONTROL_DECIMATION = 10
CONTROL_FREQUENCY_HZ = 1 / (SIMULATION_TIMESTEP * CONTROL_DECIMATION)

# Test Limits Mode Specific Amplitudes (remain static for now)
TEST_MOTION_AMPLITUDE_X = 0.05
TEST_MOTION_AMPLITUDE_Y = 0.05
TEST_MOTION_AMPLITUDE_Z = 0.035
TEST_MOTION_AMPLITUDE_ORI = np.deg2rad(30)
TEST_MOTION_FREQUENCY_HZ = 0.5 # Static frequency for test limits

# Common Config
NEUTRAL_POSITION = np.array([0.0, 0.0, 0.155 - 0.0075])
NEUTRAL_EULER_ANGLES = np.array([0.0, 0.0, 0.0])
ACTUATED_JOINT_NAMES = ["1", "2", "3", "4", "5", "6"]


# --- Plotting and Metrics (for test_limits mode) ---
def plot_joint_tracking(time_vec, commanded_q, actual_q, dof_name, joint_names_plot, output_path):
    num_joints = commanded_q.shape[1]
    fig, axs = plt.subplots(num_joints, 1, figsize=(12, 2 * num_joints), sharex=True)
    if num_joints == 1: axs = [axs]
    for i in range(num_joints):
        axs[i].plot(time_vec, commanded_q[:, i], label=f'Commanded {joint_names_plot[i]}')
        axs[i].plot(time_vec, actual_q[:, i], label=f'Actual {joint_names_plot[i]}', linestyle='--')
        axs[i].set_ylabel('Joint Angle (rad)'); axs[i].legend(); axs[i].grid(True)
    axs[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Joint Tracking Performance for {dof_name} Oscillation', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(output_path); plt.close(fig)
    print(f"Saved plot to {output_path}")

def calculate_tracking_error(commanded_q, actual_q):
    if commanded_q.shape[0] != actual_q.shape[0]:
        min_len = min(commanded_q.shape[0], actual_q.shape[0])
        commanded_q, actual_q = commanded_q[:min_len], actual_q[:min_len]
        print(f"Warning: Histories truncated to {min_len}.")
    error = commanded_q - actual_q
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(error**2, axis=0))
    return mae, rmse

# --- Test Limits Mode Logic ---
def run_oscillation_test_experiment(model, data, viewer, placo_ik,
                               dof_to_test, base_pos, base_ori,
                               amplitude_pos_x, amplitude_pos_y, amplitude_pos_z, amplitude_ori,
                               freq_hz, duration_s): # duration_s now from args
    print(f"\n--- Running Test: {dof_to_test} Oscillation ({duration_s:.1f}s) ---")
    commanded_joint_history, actual_joint_history, time_history = [], [], []
    mujoco.mj_resetData(model, data)
    sim_steps, control_steps = 0, 0
    exp_start_wall_time = time.time()
    current_motion_time = 0.0

    while current_motion_time < duration_s:
        loop_start_time = time.time()
        current_motion_time = time.time() - exp_start_wall_time
        if sim_steps % CONTROL_DECIMATION == 0:
            pose_params = {"position": base_pos.copy(), "euler_angles": base_ori.copy()}
            osc_val = np.sin(2 * np.pi * freq_hz * current_motion_time)

            if dof_to_test == "X": pose_params["position"][0] += amplitude_pos_x * osc_val
            elif dof_to_test == "Y": pose_params["position"][1] += amplitude_pos_y * osc_val
            elif dof_to_test == "Z": pose_params["position"][2] += amplitude_pos_z * osc_val
            elif dof_to_test == "Roll": pose_params["euler_angles"][0] += amplitude_ori * osc_val
            elif dof_to_test == "Pitch": pose_params["euler_angles"][1] += amplitude_ori * osc_val
            elif dof_to_test == "Yaw": pose_params["euler_angles"][2] += amplitude_ori * osc_val

            target_matrix = get_homogeneous_matrix_from_euler(**pose_params)
            try:
                target_angles = placo_ik.ik(target_matrix)
            except Exception as e:
                # print(f"IK Error @ {current_motion_time:.2f}s: {e}")
                if commanded_joint_history: target_angles = commanded_joint_history[-1]
                else: continue
            data.ctrl[:] = target_angles
            commanded_joint_history.append(target_angles.copy())
            actual_joint_history.append(get_joints(model, data).copy())
            time_history.append(current_motion_time)
            control_steps +=1
        mujoco.mj_step(model, data)
        sim_steps += 1
        if viewer and viewer.is_running(): viewer.sync()
        time.sleep(max(0, SIMULATION_TIMESTEP - (time.time() - loop_start_time)))
    print(f"Test {dof_to_test} done. {control_steps} control steps.")
    return (np.array(time_history), np.array(commanded_joint_history), np.array(actual_joint_history))

# --- Dance Mode Logic ---
def run_single_dance_move(model, data, viewer, placo_ik, move_name, duration_s, bpm, amplitude_scale):
    if move_name not in AVAILABLE_DANCE_MOVES:
        print(f"Error: Dance move '{move_name}' not found.")
        print(f"Available moves: {list(AVAILABLE_DANCE_MOVES.keys())}")
        return False # Indicate failure

    print(f"\n--- Dancing: {move_name} ({duration_s:.1f}s @ {bpm} BPM) ---")

    mujoco.mj_resetData(model, data) # Reset for each dance move segment
    neutral_matrix = get_homogeneous_matrix_from_euler(position=NEUTRAL_POSITION, euler_angles=NEUTRAL_EULER_ANGLES)
    try:
        data.ctrl[:] = placo_ik.ik(neutral_matrix)
        for _ in range(int(0.2 / SIMULATION_TIMESTEP)): mujoco.mj_step(model, data)
    except Exception as e:
        print(f"Initial IK for dance '{move_name}' failed: {e}"); data.ctrl[:] = 0.0

    sim_steps = 0
    dance_start_wall_time = time.time()

    move_function = AVAILABLE_DANCE_MOVES[move_name]
    move_params_template = get_move_parameters(move_name)
    scaled_move_params = {k: v * amplitude_scale for k, v in move_params_template.items() if isinstance(v, (int, float))}

    running = True
    current_dance_time = 0.0
    while running and current_dance_time < duration_s :
        loop_start_time = time.time()
        current_dance_time = time.time() - dance_start_wall_time
        musical_beat_time = current_dance_time * (bpm / 60.0)

        if sim_steps % CONTROL_DECIMATION == 0:
            try:
                offsets = move_function(musical_beat_time, **scaled_move_params)
            except TypeError:
                offsets = move_function(musical_beat_time)

            target_pos = NEUTRAL_POSITION + offsets.get("position_offset", np.zeros(3))
            target_ori = NEUTRAL_EULER_ANGLES + offsets.get("orientation_offset", np.zeros(3))
            target_matrix = get_homogeneous_matrix_from_euler(position=target_pos, euler_angles=target_ori)

            try:
                target_angles = placo_ik.ik(target_matrix)
                data.ctrl[:] = target_angles
            except Exception as e:
                # print(f"Dance IK Error for {move_name} @ beat {musical_beat_time:.2f}: {e}")
                pass

        mujoco.mj_step(model, data)
        sim_steps += 1
        if viewer and viewer.is_running():
            viewer.sync()
        elif viewer is None: # Headless dance mode
             pass
        else: # Viewer closed
            running = False

        elapsed_loop_time = time.time() - loop_start_time
        time.sleep(max(0, SIMULATION_TIMESTEP - elapsed_loop_time))

    print(f"Finished dancing: {move_name}")
    return True # Indicate success

# --- Argument Parser ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Stewart Platform Control Script for Testing Limits or Dancing.")
    parser.add_argument("--mode", type=str, choices=["test_limits", "dance"], required=True,
                        help="Operating mode: 'test_limits' or 'dance'.")
    parser.add_argument("--duration", type=float,
                        help="Duration in seconds. For 'test_limits', applies to each DoF test. "
                             "For 'dance' with --dance_name, total duration. "
                             "For 'dance' with --all_dances, duration PER dance move.")
    parser.add_argument("--headless", action="store_true",
                        help="Run without launching the MuJoCo viewer (useful for dance mode on servers).")

    # Dance mode specific arguments
    dance_group = parser.add_argument_group('Dance Mode Options')
    dance_selection_group = dance_group.add_mutually_exclusive_group()
    dance_selection_group.add_argument("--dance_name", type=str, default=None,
                                       choices=list(AVAILABLE_DANCE_MOVES.keys()) + [None],
                                       help="Name of a single dance move to perform.")
    dance_selection_group.add_argument("--all_dances", action="store_true",
                                       help="Perform all available dance moves sequentially.")
    dance_group.add_argument("--bpm", type=float, default=DEFAULT_DANCE_BPM,
                             help=f"Beats Per Minute for dance mode (default: {DEFAULT_DANCE_BPM}).")
    dance_group.add_argument("--amplitude_scale", type=float, default=DEFAULT_DANCE_GLOBAL_AMPLITUDE_SCALE,
                             help="Global scaling factor for dance move amplitudes "
                                  f"(default: {DEFAULT_DANCE_GLOBAL_AMPLITUDE_SCALE}).")

    args = parser.parse_args()

    # Post-parsing validation and default setting for duration
    if args.mode == "test_limits":
        if args.duration is None:
            args.duration = DEFAULT_TEST_DURATION_S
    elif args.mode == "dance":
        if not args.dance_name and not args.all_dances:
            parser.error("For dance mode, you must specify either --dance_name or --all_dances.")
        if args.duration is None:
            args.duration = DEFAULT_DANCE_DURATION_PER_MOVE_S
            if args.dance_name: # If single dance, duration is total
                print(f"No --duration specified for single dance, using default total: {args.duration}s")
            else: # if --all_dances, duration is per move
                print(f"No --duration specified for all_dances, using default per move: {args.duration}s")


    return args

# --- Main Script ---
def main(args):
    print(f"Selected Mode: {args.mode}")
    if args.headless:
        print("Running in headless mode.")

    print("Initializing MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = SIMULATION_TIMESTEP

    print("Initializing PlacoIK...")
    try:
        placo_ik = PlacoIK(PLACO_IK_DESCR_PATH)
    except Exception as e:
        print(f"Error initializing PlacoIK: {e}"); exit()

    neutral_matrix = get_homogeneous_matrix_from_euler(
        position=NEUTRAL_POSITION, euler_angles=NEUTRAL_EULER_ANGLES)
    try:
        initial_angles = placo_ik.ik(neutral_matrix)
        data.ctrl[:] = initial_angles
        for _ in range(int(0.1 / SIMULATION_TIMESTEP)): mujoco.mj_step(model, data)
    except Exception as e:
        print(f"Initial IK failed: {e}. Starting with zero controls."); data.ctrl[:] = 0.0

    print(f"Control frequency: {CONTROL_FREQUENCY_HZ:.2f} Hz")

    viewer = None
    if not args.headless:
        try:
            print("Launching MuJoCo viewer...")
            viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
            time.sleep(1)
        except Exception as e:
            print(f"Could not launch viewer: {e}. Will attempt to run headless if applicable.")
            if args.mode == "test_limits":
                print("Viewer required for test_limits mode visual feedback. Exiting if viewer failed.")
                return # test_limits really benefits from viewer for observation

    if args.mode == "test_limits":
        if not viewer and not args.headless: # Viewer failed, but headless not requested
            print("Viewer launch failed and test_limits mode selected. This mode is best with a viewer.")
            # Decide if you want to proceed or exit. For now, let's allow it.
            print("Proceeding without viewer for test_limits, plots will still be generated.")

        dofs_to_test = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
        all_metrics = {}
        for dof in dofs_to_test:
            if viewer and not viewer.is_running(): print("Viewer closed, exiting test_limits."); break

            time_vec, cmd_q, act_q = run_oscillation_test_experiment(
                model, data, viewer, placo_ik, dof, NEUTRAL_POSITION, NEUTRAL_EULER_ANGLES,
                TEST_MOTION_AMPLITUDE_X, TEST_MOTION_AMPLITUDE_Y, TEST_MOTION_AMPLITUDE_Z, TEST_MOTION_AMPLITUDE_ORI,
                TEST_MOTION_FREQUENCY_HZ, args.duration # Use duration from CLI
            )
            if time_vec.size == 0: print(f"Skipping {dof} due to empty data."); continue
            mae, rmse = calculate_tracking_error(cmd_q, act_q)
            all_metrics[dof] = {"MAE": mae, "RMSE": rmse}
            # ... (plotting and metrics printing as before) ...
            print(f"\nMetrics for {dof}:")
            for i, name in enumerate(ACTUATED_JOINT_NAMES):
                print(f"  Joint {name}: MAE = {mae[i]:.4f} rad, RMSE = {rmse[i]:.4f} rad")
            print(f"  Overall Avg MAE: {np.mean(mae):.4f} rad, RMSE: {np.mean(rmse):.4f} rad")
            plot_file = OUTPUT_DIR / f"tracking_{dof.lower()}_oscillation.png"
            plot_joint_tracking(time_vec, cmd_q, act_q, dof, ACTUATED_JOINT_NAMES, plot_file)

            if viewer and viewer.is_running(): time.sleep(0.5)
            elif not viewer and not args.headless: # if viewer failed for test_limits, short pause
                 time.sleep(0.1)


        print("\n--- Test Limits Summary ---")
        for dof, metrics in all_metrics.items():
            print(f"DoF: {dof}: Avg MAE: {np.mean(metrics['MAE']):.4f}, Avg RMSE: {np.mean(metrics['RMSE']):.4f}")

    elif args.mode == "dance":
        if args.all_dances:
            print(f"Performing all {len(AVAILABLE_DANCE_MOVES)} dances, each for {args.duration:.1f}s.")
            for move_name in AVAILABLE_DANCE_MOVES.keys():
                success = run_single_dance_move(model, data, viewer, placo_ik,
                                                move_name, args.duration, args.bpm, args.amplitude_scale)
                if not success: break # Stop if a move is not found (should not happen with keys())
                if viewer and not viewer.is_running(): print("Viewer closed during all_dances sequence."); break
                if viewer and viewer.is_running(): time.sleep(1.0) # Pause between dances if viewer is up
                elif not viewer and not args.headless: time.sleep(0.2)


        elif args.dance_name:
            run_single_dance_move(model, data, viewer, placo_ik,
                                  args.dance_name, args.duration, args.bpm, args.amplitude_scale)

    if viewer and viewer.is_running():
        print("\nProgram finished. Close the viewer to fully exit.")
        while viewer.is_running(): time.sleep(0.1)
    else:
        print("\nProgram finished.")

    if viewer: viewer.close()

if __name__ == "__main__":
    cli_args = parse_arguments()
    main(cli_args)
