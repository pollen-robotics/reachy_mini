#!/usr/bin/env python3
"""Interactive Dance Move Tester and Choreography Player for Reachy Mini.

---------------------------------------------
This script allows for real-time testing of dance moves and can also play
pre-defined choreographies from JSON files.

interactive Mode (default):
    python dance_demo.py
    - Cycles through all available moves.

Player Mode:
    python dance_demo.py --choreography choreographies/my_choreo.json
    - Plays a specific, ordered sequence of moves from a file.
"""

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini

# Correct import path as specified.
from reachy_mini.utils.rhythmic_motion import (
    AVAILABLE_DANCE_MOVES,
    MOVE_SPECIFIC_PARAMS,
)


# --- Configuration ---
@dataclass
class Config:
    """Store configuration for the dance tester."""

    bpm: float = 120.0
    control_ts: float = 0.01  # 100 Hz control loop
    beats_per_sequence: int = 8  # Switch move every 8 beats
    start_move: str = "simple_nod"
    amplitude_scale: float = 1.0
    neutral_pos: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0.02]))
    neutral_eul: np.ndarray = field(default_factory=lambda: np.zeros(3))
    choreography_path: Optional[str] = None


# --- Constants for UI ---
INTERACTIVE_HELP_MESSAGE = """
┌────────────────────────────────────────────────────────────────────────────┐
│                           INTERACTIVE CONTROLS                             │
├──────────────────────────────────┬─────────────────────────────────────────┤
│ Q / Ctrl+C : Quit Application    │ P / Space : Pause / Resume Motion       │
│ Left/Right : Previous/Next Move  │ Up/Down   : Decrease / Increase BPM     │
│ W          : Cycle Waveform      │ + / -     : Increase / Decrease Amplitude │
└──────────────────────────────────┴─────────────────────────────────────────┘
"""

CHOREO_HELP_MESSAGE = """
┌────────────────────────────────────────────────────────────────────────────┐
│                         CHOREOGRAPHY CONTROLS                              │
├──────────────────────────────────┬─────────────────────────────────────────┤
│ Q / Ctrl+C : Quit Application    │ P / Space : Pause / Resume Motion       │
│ Left/Right : Prev/Next Choreo Step │ Up/Down   : Decrease / Increase BPM   │
│ W          : Cycle Waveform      │ + / -     : Increase / Decrease Amplitude │
└──────────────────────────────────┴─────────────────────────────────────────┘
"""


# --- Shared State for Thread Communication ---
class SharedState:
    """Manage state shared between the main loop and keyboard listener thread."""

    def __init__(self) -> None:
        """Initialize the shared state."""
        self.lock = threading.Lock()
        self.running: bool = True
        self.next_move: bool = False
        self.prev_move: bool = False
        self.next_waveform: bool = False
        self.bpm_change: float = 0.0
        self.amplitude_change: float = 0.0

    def toggle_pause(self) -> bool:
        """Toggle the running state and return the new state."""
        with self.lock:
            self.running = not self.running
        return self.running

    def trigger_next_move(self) -> None:
        """Set a flag to switch to the next move/step."""
        with self.lock:
            self.next_move = True

    def trigger_prev_move(self) -> None:
        """Set a flag to switch to the previous move/step."""
        with self.lock:
            self.prev_move = True

    def trigger_next_waveform(self) -> None:
        """Set a flag to cycle to the next waveform."""
        with self.lock:
            self.next_waveform = True

    def adjust_bpm(self, amount: float) -> None:
        """Adjust the BPM by a given amount."""
        with self.lock:
            self.bpm_change += amount

    def adjust_amplitude(self, amount: float) -> None:
        """Adjust the amplitude scale by a given amount."""
        with self.lock:
            self.amplitude_change += amount

    def get_and_clear_changes(self) -> Dict[str, Any]:
        """Atomically retrieve all changes and reset them."""
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
    """Create a 4x4 homogenous transformation matrix for the head."""
    m = np.eye(4)
    m[:3, 3] = pos
    m[:3, :3] = R.from_euler("xyz", eul).as_matrix()
    return m


def keyboard_listener_thread(
    shared_state: SharedState, stop_event: threading.Event
) -> None:
    """Listen for keyboard input and update the shared state."""

    def on_press(key: Any) -> Optional[bool]:
        """Handle a key press event."""
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
        return None

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def load_choreography(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """Load a choreography from a JSON file."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: Choreography file not found at '{file_path}'")
        return None
    try:
        with open(path) as f:
            choreography = json.load(f)
        for step in choreography:
            if step.get("move") not in AVAILABLE_DANCE_MOVES:
                print(
                    f"Error: Move '{step.get('move')}' in choreography is not a valid move."
                )
                return None
        return choreography
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'")
        return None


# --- Main Application Logic ---
def main(config: Config) -> None:
    """Run the main application loop for the dance tester."""
    shared_state = SharedState()
    stop_event = threading.Event()

    choreography = None
    choreography_mode = False
    if config.choreography_path:
        choreography = load_choreography(config.choreography_path)
        if choreography is None:
            return
        choreography_mode = True

    threading.Thread(
        target=keyboard_listener_thread, args=(shared_state, stop_event), daemon=True
    ).start()

    move_names: List[str] = list(AVAILABLE_DANCE_MOVES.keys())
    waveforms: List[str] = ["sin", "cos", "triangle", "square", "sawtooth"]

    try:
        current_move_idx = move_names.index(config.start_move)
    except ValueError:
        print(
            f"Warning: Start move '{config.start_move}' not found. Starting with the first move."
        )
        current_move_idx = 0
    current_waveform_idx = 0

    t_beats, sequence_beat_counter = 0.0, 0.0
    choreography_step_idx, move_cycle_counter = 0, 0.0
    last_status_print_time, last_help_print_time = 0.0, 0.0
    bpm, amplitude_scale = config.bpm, config.amplitude_scale

    print("Connecting to Reachy Mini...")
    try:
        with ReachyMini() as mini:
            mode_text = (
                "Choreography Player" if choreography_mode else "Interactive Tester"
            )
            print(f"Robot connected. Starting {mode_text}...")
            mini.wake_up()

            print(
                CHOREO_HELP_MESSAGE if choreography_mode else INTERACTIVE_HELP_MESSAGE
            )
            last_help_print_time = time.time()

            # This prevents the first dt from including all the setup time.
            last_loop_time = time.time()

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
                if changes["next_waveform"]:
                    current_waveform_idx = (current_waveform_idx + 1) % len(waveforms)

                if choreography_mode:
                    if changes["next_move"]:
                        choreography_step_idx = (choreography_step_idx + 1) % len(
                            choreography
                        )
                        move_cycle_counter = 0.0
                    if changes["prev_move"]:
                        choreography_step_idx = (
                            choreography_step_idx - 1 + len(choreography)
                        ) % len(choreography)
                        move_cycle_counter = 0.0
                else:
                    if changes["next_move"]:
                        current_move_idx = (current_move_idx + 1) % len(move_names)
                        sequence_beat_counter = 0
                    if changes["prev_move"]:
                        current_move_idx = (
                            current_move_idx - 1 + len(move_names)
                        ) % len(move_names)
                        sequence_beat_counter = 0

                if not shared_state.running:
                    mini.set_target(
                        head_pose(config.neutral_pos, config.neutral_eul),
                        antennas=np.zeros(2),
                    )
                    time.sleep(config.control_ts)
                    continue

                beats_this_frame = dt * (bpm / 60.0)

                if choreography_mode:
                    current_step = choreography[choreography_step_idx]
                    target_cycles = current_step["cycles"]
                    move_cycle_counter += beats_this_frame

                    if move_cycle_counter >= target_cycles:
                        if choreography_step_idx == len(choreography) - 1:
                            print("\nChoreography complete. Exiting.")
                            break
                        choreography_step_idx += 1
                        move_cycle_counter = 0.0
                        # Re-fetch the current step in case we just advanced
                        current_step = choreography[choreography_step_idx]

                    move_name = current_step["move"]
                    step_amplitude_modifier = current_step.get("amplitude", 1.0)
                    t_motion = move_cycle_counter
                else:
                    sequence_beat_counter += beats_this_frame
                    if sequence_beat_counter >= config.beats_per_sequence:
                        current_move_idx = (current_move_idx + 1) % len(move_names)
                        sequence_beat_counter = 0

                    move_name = move_names[current_move_idx]
                    step_amplitude_modifier = 1.0
                    t_beats += beats_this_frame
                    t_motion = t_beats

                waveform = waveforms[current_waveform_idx]
                move_fn = AVAILABLE_DANCE_MOVES[move_name]
                base_params = MOVE_SPECIFIC_PARAMS.get(move_name, {}).copy()
                current_params = base_params

                if "waveform" in base_params:
                    current_params["waveform"] = waveform

                final_amplitude_scale = amplitude_scale * step_amplitude_modifier
                for key in current_params:
                    if "amplitude" in key or "_amp" in key:
                        current_params[key] *= final_amplitude_scale

                offsets = move_fn(t_motion, **current_params)

                final_pos = config.neutral_pos + offsets.position_offset
                final_eul = config.neutral_eul + offsets.orientation_offset
                final_ant = offsets.antennas_offset
                mini.set_target(head_pose(final_pos, final_eul), antennas=final_ant)

                if loop_start_time - last_status_print_time > 1.0:
                    sys.stdout.write("\r" + " " * 80 + "\r")
                    status = "RUNNING" if shared_state.running else "PAUSED "

                    if choreography_mode:
                        target_cycles_display = choreography[choreography_step_idx][
                            "cycles"
                        ]
                        progress_pct = (
                            f"{(move_cycle_counter / target_cycles_display * 100):.0f}%"
                        )
                        status_line = (
                            f"[{status}] Step {choreography_step_idx + 1}/{len(choreography)}: {move_name:<20} ({progress_pct:>4}) | "
                            f"BPM: {bpm:<5.1f} | Amp: {final_amplitude_scale:.1f}x"
                        )
                    else:
                        wave_status = (
                            waveform if "waveform" in current_params else "N/A"
                        )
                        status_line = f"[{status}] Move: {move_name:<35} | BPM: {bpm:<5.1f} | Wave: {wave_status:<8} | Amp: {amplitude_scale:.1f}x"
                    print(status_line, end="")
                    sys.stdout.flush()
                    last_status_print_time = loop_start_time

                if loop_start_time - last_help_print_time > 30.0:
                    print(
                        f"\n{CHOREO_HELP_MESSAGE if choreography_mode else INTERACTIVE_HELP_MESSAGE}"
                    )
                    last_help_print_time = loop_start_time

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
        description="Interactive Dance Move Tester and Choreography Player for Reachy Mini.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bpm", type=float, default=120.0, help="Starting BPM.")
    parser.add_argument(
        "--start-move",
        default="simple_nod",
        choices=list(AVAILABLE_DANCE_MOVES.keys()),
        help="Which dance move to start with in interactive mode.",
    )
    parser.add_argument(
        "--beats-per-sequence",
        type=int,
        default=16,
        help="In interactive mode, automatically change move after this many beats.",
    )
    parser.add_argument(
        "--choreography",
        type=str,
        default=None,
        help="Path to a JSON choreography file to play. Overrides interactive mode.",
    )

    cli_args = parser.parse_args()
    app_config = Config(
        bpm=cli_args.bpm,
        start_move=cli_args.start_move,
        beats_per_sequence=cli_args.beats_per_sequence,
        choreography_path=cli_args.choreography,
    )

    main(app_config)
