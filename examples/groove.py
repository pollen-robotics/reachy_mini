import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse

# --- SDK Import ---
from reachy_mini import ReachyMini # Assuming SDK is installed

# --- Dance Library Import ---
# Assumes 'dance_moves_library.py' is in the same directory or Python path
from dance_moves import (
    AVAILABLE_DANCE_MOVES,
    get_move_parameters
)

# --- Default Configuration ---
DEFAULT_DANCE_DURATION_PER_MOVE_S = 3.0
DEFAULT_DANCE_BPM = 120.0
CONTROL_TIMESTEP = 0.01  # control loop period in seconds

# --- Robot Configuration ---
NEUTRAL_HEAD_POSITION = np.array([0.0, 0.0, 0.177 - 0.0075]) # Neutral head position in meters (x, y, z

NEUTRAL_HEAD_EULER_ANGLES = np.array([0.0, 0.0, 0.0]) # Roll, Pitch, Yaw in radians

def create_head_pose(position: np.ndarray, euler_angles: np.ndarray) -> np.ndarray:
    """
    Create a 4x4 pose matrix for the head given position and Euler angles.
    
    :param position: 3D position vector (x, y, z).
    :param euler_angles: Euler angles in radians (roll, pitch, yaw).
    :return: 4x4 pose matrix.
    """
    pose_matrix = np.eye(4)
    pose_matrix[:3, 3] = position
    rotation_matrix = R.from_euler('xyz', euler_angles, degrees=False).as_matrix()
    pose_matrix[:3, :3] = rotation_matrix
    return pose_matrix

# --- Dance Execution Logic ---
def run_dance_move(reachy_mini: ReachyMini, move_name: str, duration_s: float, bpm: float, perfect_ending: bool = False) -> bool:
    if move_name not in AVAILABLE_DANCE_MOVES:
        print(f"Error: Dance move '{move_name}' not found.")
        print(f"Available moves: {list(AVAILABLE_DANCE_MOVES.keys())}")
        return False

    print(f"\n--- Dancing: {move_name} ({duration_s:.1f}s @ {bpm} BPM) ---")

    move_function = AVAILABLE_DANCE_MOVES[move_name]
    current_move_params = get_move_parameters(move_name)
    if perfect_ending:
        # duration will be slightly longer to ensure the movement ends where it started
        frequency_factor = current_move_params.get("frequency_factor", 1.0)
        move_period = 60.0 / bpm / frequency_factor
        duration_s = move_period * np.ceil(duration_s / move_period)

    dance_start_wall_time = time.time()
    current_dance_time = 0.0

    while current_dance_time < duration_s:
        loop_start_time = time.time()
        current_dance_time = time.time() - dance_start_wall_time
        musical_beat_time = current_dance_time * (bpm / 60.0)
        offsets = move_function(musical_beat_time, **current_move_params)

        pos_offset = offsets.get("position_offset", np.zeros(3))
        orient_offset_euler = offsets.get("orientation_offset", np.zeros(3))
        antennas_command = offsets.get("antennas_offset", np.zeros(2))

        # Calculate target pose
        head_pose = create_head_pose(NEUTRAL_HEAD_POSITION + pos_offset, NEUTRAL_HEAD_EULER_ANGLES + orient_offset_euler)
        
        reachy_mini.set_position(head=head_pose, antennas=antennas_command)
        
        # Maintain control loop frequency
        elapsed_loop_time = time.time() - loop_start_time
        sleep_time = CONTROL_TIMESTEP - elapsed_loop_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print(f"Warning: Loop for {move_name} took too long: {elapsed_loop_time:.4f}s")


    print(f"Finished dancing: {move_name}")
    return True

# --- Argument Parser ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Reachy Mini Dance Script")
    # Making the group not required, so default behavior can be all_dances
    dance_selection_group = parser.add_mutually_exclusive_group(required=False)
    dance_selection_group.add_argument("--dance_name", type=str,
                                       choices=list(AVAILABLE_DANCE_MOVES.keys()),
                                       help="Name of a single dance move to perform.")
    dance_selection_group.add_argument("--all_dances", action="store_true",
                                       help="Perform all available dance moves sequentially (default if no other option is chosen).")

    parser.add_argument("--duration", type=float, default=DEFAULT_DANCE_DURATION_PER_MOVE_S,
                        help=f"Duration in seconds PER dance move (default: {DEFAULT_DANCE_DURATION_PER_MOVE_S}s).")
    parser.add_argument("--bpm", type=float, default=DEFAULT_DANCE_BPM,
                        help=f"Beats Per Minute for the dance (default: {DEFAULT_DANCE_BPM}).")

    args = parser.parse_args()

    # If no dance selection argument is provided, default to all_dances
    if not args.dance_name and not args.all_dances:
        args.all_dances = True
        print("No specific dance chosen, defaulting to --all_dances.")

    return args

# --- Main Script ---
def main():
    args = parse_arguments()
    reachy_mini = None
    neutral_matrix = create_head_pose(NEUTRAL_HEAD_POSITION, NEUTRAL_HEAD_EULER_ANGLES)

    try:
        print("Attempting to connect to Reachy Mini...")
        reachy_mini = ReachyMini()
        print("Successfully connected to Reachy Mini.")

        print("Moving to neutral pose...")
        reachy_mini.set_position(head=neutral_matrix, antennas=np.array([0.0, 0.0]))
        time.sleep(0.5)

        if args.all_dances:
            print(f"Performing all {len(AVAILABLE_DANCE_MOVES)} dances, each for {args.duration:.1f}s.")
            dance_names = list(AVAILABLE_DANCE_MOVES.keys())
            for i, move_name in enumerate(dance_names):
                run_dance_move(reachy_mini, move_name, args.duration, args.bpm, perfect_ending=True)
        elif args.dance_name:
            run_dance_move(reachy_mini, args.dance_name, args.duration, args.bpm)

        print("\nDance sequence finished.")

    except KeyboardInterrupt:
        print("\nDance interrupted by user. Exiting.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if reachy_mini is not None:
            print("Program ended, returning to neutral pose...")
            reachy_mini.set_position(head=neutral_matrix, antennas=np.array([0.0, 0.0]))
            time.sleep(1)


if __name__ == "__main__":
    main()