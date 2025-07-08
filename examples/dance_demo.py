#!/usr/bin/env python3
# dance_tester.py
"""Interactive Dance Move Tester for Reachy Mini.

---------------------------------------------
This script allows for real-time testing and exploration of dance moves from
the `dance_moves` library on a Reachy Mini robot. It simulates a constant BPM,
allowing you to focus on the motion itself.

Controls are displayed on-screen and on the robot's orb display.
"""

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini
from reachy_mini.moves.dance_moves import (
    AVAILABLE_DANCE_MOVES,
    MOVE_SPECIFIC_PARAMS,
)


# --- Configuration ---
@dataclass
class Config:
    """Configuration for the dance tester."""

    bpm: float = 120.0
    control_ts: float = 0.01  # 100 Hz control loop
    beats_per_sequence: int = 8  # Switch move every 8 beats
    start_move: str = "simple_nod"
    amplitude_scale: float = 1.0
    neutral_pos: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0.02]))
    neutral_eul: np.ndarray = field(default_factory=lambda: np.zeros(3))


# --- Constants for UI ---
HELP_MESSAGE = """
┌────────────────────────────────────────────────────────────────────────────┐
│                              CONTROLS                                      │
├──────────────────────────────────┬─────────────────────────────────────────┤
│ Q / Ctrl+C : Quit Application    │ P / Space : Pause / Resume Motion       │
│ Left/Right : Previous/Next Move  │ Up/Down   : Decrease / Increase BPM     │
│ W          : Cycle Waveform      │ + / -     : Increase / Decrease Amplitude │
└──────────────────────────────────┴─────────────────────────────────────────┘
"""

ORB_HELP_LINES = [
    "Controls:",
    "Arrows -> Next Move",
    "Up/Down -> BPM",
    "W -> Waveform",
    "+ / - -> Amplitude",
    "Space/P -> Pause",
    "Q -> Quit",
]


# --- Shared State for Thread Communication ---
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.next_move = False
        self.prev_move = False
        self.next_waveform = False
        self.bpm_change = 0.0
        self.amplitude_change = 0.0

    def toggle_pause(self):
        with self.lock:
            self.running = not self.running
        return self.running

    def trigger_next_move(self):
        with self.lock:
            self.next_move = True

    def trigger_prev_move(self):
        with self.lock:
            self.prev_move = True

    def trigger_next_waveform(self):
        with self.lock:
            self.next_waveform = True

    def adjust_bpm(self, amount: float):
        with self.lock:
            self.bpm_change += amount

    def adjust_amplitude(self, amount: float):
        with self.lock:
            self.amplitude_change += amount

    def get_and_clear_changes(self) -> dict:
        with self.lock:
            changes = {
                "next_move": self.next_move,
                "prev_move": self.prev_move,
                "next_waveform": self.next_waveform,
                "bpm_change": self.bpm_change,
                "amplitude_change": self.amplitude_change,
            }
            self.next_move = self.prev_move = self.next_waveform = False
            self.bpm_change = self.amplitude_change = 0.0
            return changes


# --- Robot Interaction & Utilities ---
def head_pose(pos: np.ndarray, eul: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = pos
    m[:3, :3] = R.from_euler("xyz", eul).as_matrix()
    return m


def keyboard_listener_thread(shared_state: SharedState, stop_event: threading.Event):
    def on_press(key):
        if stop_event.is_set():
            return False
        if hasattr(key, "char"):
            if key.char.lower() == "q":
                stop_event.set()
                return False
            if key.char.lower() == "p":
                shared_state.toggle_pause()
            if key.char.lower() == "w":
                shared_state.trigger_next_waveform()
            if key.char == "+":
                shared_state.adjust_amplitude(0.1)
            if key.char == "-":
                shared_state.adjust_amplitude(-0.1)
        if key == keyboard.Key.space:
            shared_state.toggle_pause()
        elif key == keyboard.Key.right:
            shared_state.trigger_next_move()
        elif key == keyboard.Key.left:
            shared_state.trigger_prev_move()
        elif key == keyboard.Key.up:
            shared_state.adjust_bpm(5.0)
        elif key == keyboard.Key.down:
            shared_state.adjust_bpm(-5.0)

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


# --- Main Application Logic ---
def main(config: Config):
    """Main function to run the dance tester."""
    shared_state = SharedState()
    stop_event = threading.Event()

    threading.Thread(
        target=keyboard_listener_thread, args=(shared_state, stop_event), daemon=True
    ).start()

    move_names = list(AVAILABLE_DANCE_MOVES.keys())
    waveforms = ["sin", "cos", "triangle", "square", "sawtooth"]

    try:
        current_move_idx = move_names.index(config.start_move)
    except ValueError:
        print(
            f"Warning: Start move '{config.start_move}' not found. Starting with the first move."
        )
        current_move_idx = 0
    current_waveform_idx = 0

    t_beats, sequence_beat_counter, last_loop_time = 0.0, 0.0, time.time()
    last_status_print_time, last_help_print_time, last_orb_update_time = 0.0, 0.0, 0.0
    bpm, amplitude_scale = config.bpm, config.amplitude_scale
    current_orb_line_idx = 0

    print("Connecting to Reachy Mini...")
    try:
        with ReachyMini() as mini:  # Changed from 'bot' to 'mini'
            print("Robot connected. Starting dance test...")
            mini.set_target(
                head_pose(config.neutral_pos, config.neutral_eul), antennas=np.zeros(2)
            )
            time.sleep(1.0)

            print(HELP_MESSAGE)  # Print controls at the start
            last_help_print_time = time.time()

            while not stop_event.is_set():
                loop_start_time = time.time()
                dt = loop_start_time - last_loop_time
                last_loop_time = loop_start_time

                changes = shared_state.get_and_clear_changes()
                if changes["bpm_change"]:
                    bpm = max(20.0, bpm + changes["bpm_change"])
                if changes["amplitude_change"]:
                    amplitude_scale = max(
                        0.1, amplitude_scale + changes["amplitude_change"]
                    )
                if changes["next_move"]:
                    current_move_idx = (current_move_idx + 1) % len(move_names)
                    sequence_beat_counter = 0
                if changes["prev_move"]:
                    current_move_idx = (current_move_idx - 1 + len(move_names)) % len(
                        move_names
                    )
                    sequence_beat_counter = 0
                if changes["next_waveform"]:
                    current_waveform_idx = (current_waveform_idx + 1) % len(waveforms)

                if not shared_state.running:
                    mini.set_target(
                        head_pose(config.neutral_pos, config.neutral_eul),
                        antennas=np.zeros(2),
                    )  # Changed to 'mini'
                    time.sleep(config.control_ts)
                    continue

                beats_this_frame = dt * (bpm / 60.0)
                t_beats += beats_this_frame
                sequence_beat_counter += beats_this_frame

                if sequence_beat_counter >= config.beats_per_sequence:
                    current_move_idx = (current_move_idx + 1) % len(move_names)
                    sequence_beat_counter = 0

                move_name = move_names[current_move_idx]
                waveform = waveforms[current_waveform_idx]
                move_fn = AVAILABLE_DANCE_MOVES[move_name]
                base_params = MOVE_SPECIFIC_PARAMS.get(move_name, {})
                current_params = base_params.copy()
                if "waveform" in move_fn.__code__.co_varnames:
                    current_params["waveform"] = waveform
                for key in current_params:
                    if "amplitude" in key:
                        current_params[key] *= amplitude_scale

                offsets = move_fn(t_beats, **current_params)

                final_pos, final_eul, final_ant = (
                    config.neutral_pos + offsets.position_offset,
                    config.neutral_eul + offsets.orientation_offset,
                    offsets.antennas_offset,
                )
                mini.set_target(
                    head_pose(final_pos, final_eul), antennas=final_ant
                )  # Changed to 'mini'

                # --- UI Update (Console and Orb) ---
                if loop_start_time - last_status_print_time > 1.0:
                    sys.stdout.write("\r" + " " * 80 + "\r")
                    status = "RUNNING" if shared_state.running else "PAUSED "
                    wave_status = waveform if "waveform" in current_params else "N/A"
                    print(
                        f"[{status}] Move: {move_name:<35} | BPM: {bpm:<5.1f} | Wave: {wave_status:<8} | Amp: {amplitude_scale:.1f}x",
                        end="",
                    )
                    sys.stdout.flush()
                    last_status_print_time = loop_start_time

                if loop_start_time - last_help_print_time > 20.0:
                    print(f"\n{HELP_MESSAGE}")  # Reprint help message periodically
                    last_help_print_time = loop_start_time

                if loop_start_time - last_orb_update_time > 4.0:
                    current_orb_line_idx = (current_orb_line_idx + 1) % len(
                        ORB_HELP_LINES
                    )
                    last_orb_update_time = loop_start_time

                time.sleep(max(0, config.control_ts - (time.time() - loop_start_time)))

    except KeyboardInterrupt:
        print("\nCtrl-C received. Shutting down...")
    except Exception as e:
        import traceback

        print(f"\nAn error occurred: {e}")
        traceback.print_exc()
    finally:
        stop_event.set()
        print("\nPutting robot to sleep and cleaning up...")
        with ReachyMini() as mini:
            mini.goto_sleep()
        print("Shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive Dance Move Tester for Reachy Mini.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bpm", type=float, default=120.0, help="Starting BPM for the simulation."
    )
    parser.add_argument(
        "--start-move",
        default="simple_nod",
        choices=AVAILABLE_DANCE_MOVES.keys(),
        help="Which dance move to start with.",
    )
    parser.add_argument(
        "--beats-per-sequence",
        type=int,
        default=16,
        help="Automatically change move after this many beats.",
    )
    cli_args = parser.parse_args()
    app_config = Config(
        bpm=cli_args.bpm,
        start_move=cli_args.start_move,
        beats_per_sequence=cli_args.beats_per_sequence,
    )
    main(app_config)
