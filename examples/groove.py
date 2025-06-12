import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse

# --- New SDK Import ---
from reachy_mini import ReachyMini

# --- Individual Move Parameters (from your dance_moves.py) ---
MOVE_SPECIFIC_PARAMS = {
    "simple_nod": {"amplitude_rad": np.deg2rad(20), "frequency_factor": 1.0},
    "head_bob_z": {"amplitude_m": 0.03, "frequency_factor": 1.0},
    "side_to_side_sway": {"amplitude_m": 0.04, "frequency_factor": 0.5},
    "head_tilt_roll": {"amplitude_rad": np.deg2rad(25), "frequency_factor": 0.5},
    "look_around_yaw": {"amplitude_rad": np.deg2rad(30), "frequency_factor": 0.25},
    "circular_head_roll": {"pitch_amplitude_rad": np.deg2rad(15), "roll_amplitude_rad": np.deg2rad(15), "frequency_factor": 0.25},
    "robot_z_bounce": {"amplitude_m": 0.035, "frequency_factor": 2.0},
    "floating_x_glide": {"amplitude_m": 0.05, "frequency_factor": 1.0},
    "side_to_side_head_bob_what_is_love": {"roll_amplitude_rad": np.deg2rad(40), "frequency_factor": 1.0}
}

def get_move_parameters(move_name):
    return MOVE_SPECIFIC_PARAMS.get(move_name, {})

# --- Dance Move Functions (from your dance_moves.py) ---
# These functions return dictionaries with "position_offset" and "orientation_offset" (Euler XYZ in radians)

def move_simple_nod(musical_beat_time, amplitude_rad=np.deg2rad(15), frequency_factor=1.0):
    pitch_offset = amplitude_rad * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    return {"position_offset": np.zeros(3), "orientation_offset": np.array([0.0, pitch_offset, 0.0])}

def move_head_bob_z(musical_beat_time, amplitude_m=0.02, frequency_factor=1.0):
    z_offset = -amplitude_m * (1 - np.cos(2 * np.pi * frequency_factor * musical_beat_time)) / 2
    return {"position_offset": np.array([0.0, 0.0, z_offset]), "orientation_offset": np.zeros(3)}

def move_side_to_side_sway(musical_beat_time, amplitude_m=0.03, frequency_factor=1.0):
    y_offset = amplitude_m * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    return {"position_offset": np.array([0.0, y_offset, 0.0]), "orientation_offset": np.zeros(3)}

def move_head_tilt_roll(musical_beat_time, amplitude_rad=np.deg2rad(20), frequency_factor=1.0):
    roll_offset = amplitude_rad * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    return {"position_offset": np.zeros(3), "orientation_offset": np.array([roll_offset, 0.0, 0.0])}

def move_look_around_yaw(musical_beat_time, amplitude_rad=np.deg2rad(25), frequency_factor=1.0):
    yaw_offset = amplitude_rad * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    return {"position_offset": np.zeros(3), "orientation_offset": np.array([0.0, 0.0, yaw_offset])}

def move_circular_head_roll(musical_beat_time, pitch_amplitude_rad=np.deg2rad(10), roll_amplitude_rad=np.deg2rad(10), frequency_factor=1.0):
    pitch_offset = pitch_amplitude_rad * np.cos(2 * np.pi * frequency_factor * musical_beat_time)
    roll_offset = roll_amplitude_rad * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    return {"position_offset": np.zeros(3), "orientation_offset": np.array([roll_offset, pitch_offset, 0.0])}

def move_robot_z_bounce(musical_beat_time, amplitude_m=0.03, frequency_factor=1.0):
    z_offset = amplitude_m * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    return {"position_offset": np.array([0.0, 0.0, z_offset]), "orientation_offset": np.zeros(3)}

def move_floating_x_glide(musical_beat_time, amplitude_m=0.04, frequency_factor=1.0):
    x_offset = amplitude_m * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    return {"position_offset": np.array([x_offset, 0.0, 0.0]), "orientation_offset": np.zeros(3)}

def move_side_to_side_head_bob_what_is_love(musical_beat_time, roll_amplitude_rad=np.deg2rad(40), frequency_factor=1.0):
    roll_offset = roll_amplitude_rad * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    return {
        "position_offset": np.zeros(3),
        "orientation_offset": np.array([roll_offset, 0.0, 0.0])
    }

# --- Library of Available Moves (from your dance_moves.py) ---
AVAILABLE_DANCE_MOVES = {
    "simple_nod": move_simple_nod,
    "head_bob_z": move_head_bob_z,
    "side_to_side_sway": move_side_to_side_sway,
    "head_tilt_roll": move_head_tilt_roll,
    "look_around_yaw": move_look_around_yaw,
    "circular_head_roll": move_circular_head_roll,
    "robot_z_bounce": move_robot_z_bounce,
    "floating_x_glide": move_floating_x_glide,
    "side_to_side_head_bob_what_is_love": move_side_to_side_head_bob_what_is_love,
}

# --- Default Configuration ---
DEFAULT_DANCE_DURATION_PER_MOVE_S = 2.0
DEFAULT_DANCE_BPM = 120.0
DEFAULT_DANCE_GLOBAL_AMPLITUDE_SCALE = 1.0
CONTROL_TIMESTEP = 0.02  # 50 Hz control loop

# --- Robot Configuration ---
# Neutral pose: position (x, y, z) and euler angles (roll, pitch, yaw in radians)
NEUTRAL_HEAD_POSITION = np.array([0.0, 0.0, 0.177- 0.0075]) # x, y, z in meters

NEUTRAL_HEAD_EULER_ANGLES = np.array([0.0, 0.0, 0.0]) # Roll, Pitch, Yaw

# --- Dance Execution Logic ---
def run_dance_move(reachy_mini: ReachyMini, move_name: str, duration_s: float, bpm: float, amplitude_scale: float):
    if move_name not in AVAILABLE_DANCE_MOVES:
        print(f"Error: Dance move '{move_name}' not found.")
        print(f"Available moves: {list(AVAILABLE_DANCE_MOVES.keys())}")
        return False

    print(f"\n--- Dancing: {move_name} ({duration_s:.1f}s @ {bpm} BPM, Scale: {amplitude_scale}) ---")

    move_function = AVAILABLE_DANCE_MOVES[move_name]
    # Get the specific parameters for this move, then scale them
    move_params_template = get_move_parameters(move_name)
    scaled_move_params = {}
    for k, v in move_params_template.items():
        if isinstance(v, (int, float)): # Only scale numerical parameters
            scaled_move_params[k] = v * amplitude_scale
        else:
            scaled_move_params[k] = v # Keep non-numerical params as is (e.g., if any)


    dance_start_wall_time = time.time()
    current_dance_time = 0.0

    while current_dance_time < duration_s:
        loop_start_time = time.time()
        current_dance_time = time.time() - dance_start_wall_time
        musical_beat_time = current_dance_time * (bpm / 60.0)

        # Get position and orientation offsets from the dance move function
        try:
            # Pass scaled parameters if the function accepts them
            offsets = move_function(musical_beat_time, **scaled_move_params)
        except TypeError:
            # Fallback if the function doesn't accept keyword arguments (e.g. expects defaults)
            # This can happen if a move in MOVE_SPECIFIC_PARAMS has no entries or
            # if the function signature doesn't match all keys in scaled_move_params
            # For simplicity, we assume if **scaled_move_params fails, it expects no extra args
            # beyond musical_beat_time and will use its internal defaults.
            # A more robust solution might involve inspecting function signature.
            print(f"Warning: Could not pass scaled params to {move_name}. Using function defaults.")
            offsets = move_function(musical_beat_time)


        pos_offset = offsets.get("position_offset", np.zeros(3))
        orient_offset_euler = offsets.get("orientation_offset", np.zeros(3))

        # Calculate target pose
        target_pos = NEUTRAL_HEAD_POSITION + pos_offset
        target_euler = NEUTRAL_HEAD_EULER_ANGLES + orient_offset_euler

        # Convert to 4x4 homogeneous matrix for the SDK
        head_pose_matrix = np.eye(4)
        head_pose_matrix[:3, 3] = target_pos
        try:
            rot_matrix = R.from_euler('xyz', target_euler, degrees=False).as_matrix()
            head_pose_matrix[:3, :3] = rot_matrix
        except Exception as e:
            print(f"Error creating rotation matrix: {e}. Skipping frame.")
            continue


        # Simple antenna movement (example: make them wiggle in opposition)
        # Antenna values are typically expected in [-1, 1] range for full movement.
        antenna_val_rad = np.sin(2 * np.pi * 0.5 * musical_beat_time) # 0.5 Hz relative to beat
        antennas_command = np.array([antenna_val_rad, -antenna_val_rad]) * 0.8 # Scale to not hit limits

        # Send to robot
        reachy_mini.set_position(head=head_pose_matrix, antennas=antennas_command)

        # Maintain control loop frequency
        elapsed_loop_time = time.time() - loop_start_time
        sleep_time = CONTROL_TIMESTEP - elapsed_loop_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    print(f"Finished dancing: {move_name}")
    return True

# --- Argument Parser ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Reachy Mini Dance Script")
    dance_selection_group = parser.add_mutually_exclusive_group(required=True)
    dance_selection_group.add_argument("--dance_name", type=str, choices=list(AVAILABLE_DANCE_MOVES.keys()),
                                       help="Name of a single dance move to perform.")
    dance_selection_group.add_argument("--all_dances", action="store_true",
                                       help="Perform all available dance moves sequentially.")

    parser.add_argument("--duration", type=float, default=DEFAULT_DANCE_DURATION_PER_MOVE_S,
                        help=f"Duration in seconds PER dance move (default: {DEFAULT_DANCE_DURATION_PER_MOVE_S}s).")
    parser.add_argument("--bpm", type=float, default=DEFAULT_DANCE_BPM,
                        help=f"Beats Per Minute for the dance (default: {DEFAULT_DANCE_BPM}).")
    parser.add_argument("--amplitude_scale", type=float, default=DEFAULT_DANCE_GLOBAL_AMPLITUDE_SCALE,
                        help="Global scaling factor for dance move amplitudes "
                             f"(default: {DEFAULT_DANCE_GLOBAL_AMPLITUDE_SCALE}).")

    args = parser.parse_args()
    return args

# --- Main Script ---
def main():
    args = parse_arguments()

    print(f"Connecting to Reachy Mini...")
    try:
        with ReachyMini() as reachy_mini:
            print("Connected to Reachy Mini.")

            # Go to a neutral pose first
            neutral_matrix = np.eye(4)
            neutral_matrix[:3, 3] = NEUTRAL_HEAD_POSITION
            neutral_rot_matrix = R.from_euler('xyz', NEUTRAL_HEAD_EULER_ANGLES, degrees=False).as_matrix()
            neutral_matrix[:3, :3] = neutral_rot_matrix
            reachy_mini.set_position(head=neutral_matrix, antennas=np.array([0.0, 0.0]))
            print("Moved to neutral pose. Waiting 2 seconds before starting...")
            time.sleep(0.5)


            if args.all_dances:
                print(f"Performing all {len(AVAILABLE_DANCE_MOVES)} dances, each for {args.duration:.1f}s.")
                for move_name in AVAILABLE_DANCE_MOVES.keys():
                    run_dance_move(reachy_mini, move_name, args.duration, args.bpm, args.amplitude_scale)
                    print("Pausing for 1 second between dances...")
                    reachy_mini.set_position(head=neutral_matrix, antennas=np.array([0.0, 0.0])) # Return to neutral
                    time.sleep(1.0)
            elif args.dance_name:
                run_dance_move(reachy_mini, args.dance_name, args.duration, args.bpm, args.amplitude_scale)

            print("\nDance sequence finished. Returning to neutral pose.")
            reachy_mini.set_position(head=neutral_matrix, antennas=np.array([0.0, 0.0]))
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nDance interrupted by user. Exiting.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Program ended.")


if __name__ == "__main__":
    main()