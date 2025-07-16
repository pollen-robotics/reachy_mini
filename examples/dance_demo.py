import json
import signal
import sys
import threading
import time
from typing import Any, Dict

import numpy as np
from pynput import keyboard

from reachy_mini import ReachyMini

# Assuming your library files are in a 'rhythmic_motion' subdirectory
from rhythmic_motion.moves import AVAILABLE_MOVES

# Global flag to signal shutdown
shutdown_flag = threading.Event()


def load_choreography(filepath: str) -> Dict[str, Any]:
    """Load the choreography sequence from a JSON file."""
    with open(filepath) as f:
        return json.load(f)


def get_base_pose(robot: ReachyMini) -> np.ndarray:
    """Get the robot's initial neck orientation as the base pose."""
    robot.neck.compliant = False
    time.sleep(0.1)
    initial_orientation_matrix = robot.neck.orientation
    robot.neck.compliant = True
    return initial_orientation_matrix


def main():
    """Main function to run the dance demo."""
    robot = ReachyMini(
        config_file="reachy_mini_config.json",
    )

    choreography = load_choreography("choreography.json")
    bpm = choreography.get("bpm", 120)
    beats_per_second = bpm / 60.0
    sequence = choreography.get("sequence", [])

    print("=" * 30)
    print(" Reachy Mini Dance Demo ".center(30, " "))
    print("=" * 30)
    print(f"BPM: {bpm}")
    print("Press 'q' or Ctrl-C to stop.")

    listener_thread = threading.Thread(target=keyboard_listener_thread)
    listener_thread.start()

    base_orientation = get_base_pose(robot)
    robot.neck.compliant = True
    start_time = time.time()
    sequence_index = 0

    try:
        while not shutdown_flag.is_set() and sequence_index < len(sequence):
            move_item = sequence[sequence_index]
            move_name = move_item["move"]
            duration_beats = move_item["cycles"]

            if move_name not in AVAILABLE_MOVES:
                print(f"Warning: Move '{move_name}' not found. Skipping.")
                sequence_index += 1
                continue

            # CHANGE 1: Unpack info and print description
            move_fn, move_params, move_info = AVAILABLE_MOVES[move_name]
            description = move_info.get("description", "No description available.")

            print(
                f"\n▶️  Playing: {move_name.replace('_', ' ').title()} ({duration_beats} beats)"
            )
            print(f"   '{description}'")

            move_start_time = time.time()
            while time.time() - move_start_time < duration_beats / beats_per_second:
                if shutdown_flag.is_set():
                    break

                elapsed_time = time.time() - start_time
                t_beats = elapsed_time * beats_per_second

                offsets = move_fn(t_beats, **move_params)

                target_orientation = base_orientation.dot(
                    ReachyMini.rpy_to_rotation_matrix(offsets.orientation_offset)
                )

                robot.neck.target_orientation = target_orientation
                robot.antennas.target_positions = offsets.antennas_offset

                time.sleep(0.02)

            sequence_index += 1

            if shutdown_flag.is_set():
                break

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        print("\nDance sequence finished or interrupted.")
        shutdown_robot(robot)
        shutdown_flag.set()
        listener_thread.join()


def on_press(key):
    """Callback function for key presses."""
    # CHANGE 2: Robust check to prevent crash on special keys
    if hasattr(key, "char") and key.char and key.char.lower() == "q":
        print("\n'q' pressed. Initiating shutdown...")
        shutdown_flag.set()
        return False


def keyboard_listener_thread():
    """Thread to listen for keyboard input."""
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
    print("Keyboard listener stopped.")


def shutdown_robot(robot: ReachyMini):
    """Gracefully shuts down the robot."""
    print("\nPutting robot to sleep and cleaning up...")
    try:
        robot.neck.compliant = True
        time.sleep(0.1)
        robot.turn_off()
    except Exception as e:
        print(f"Error during robot shutdown: {e}")
    print("Shutdown complete.")


def signal_handler(sig, frame):
    """Handle Ctrl-C signal."""
    print("\nCtrl-C received. Shutting down...")
    shutdown_flag.set()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()
