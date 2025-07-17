#!/usr/bin/env python3
"""Mini-script to manually find the Z-coordinate limits of Reachy Mini's head.

This script allows you to directly control the vertical (Z-axis) position
of the robot's head using your keyboard's arrow keys. The current Z-value
being sent to the robot is printed to the console in real-time.

This is useful for manually determining the safe and reachable minimum and
maximum vertical positions for creating new dance moves or poses.

CONTROLS:
- UP ARROW   : Increase Z-coordinate (move head up)
- DOWN ARROW : Decrease Z-coordinate (move head down)
- Q or ESC   : Quit the application
"""

import sys
import threading
import time
from typing import Any

import numpy as np
from pynput import keyboard

from reachy_mini import ReachyMini, utils

# --- Configuration ---
# Starting Z position in meters. 0.02 is a safe neutral height.
Z_START = 0.02
# How much to change the Z-coordinate with each key press (in meters).
Z_INCREMENT = 0.005  # 5 millimeters
# The rate at which commands are sent to the robot (in seconds).
CONTROL_LOOP_RATE = 0.02  # 50 Hz


# --- Shared State for Thread Communication ---
# A simple class to hold state that the keyboard listener thread will modify.
class ControlState:
    """Manages state shared between the main loop and keyboard listener."""

    def __init__(self, initial_z: float):
        self.z_coord: float = initial_z
        self.running: bool = True
        self.lock = threading.Lock()

    def adjust_z(self, amount: float) -> None:
        """Safely adjust the Z-coordinate."""
        with self.lock:
            self.z_coord += amount

    def get_z(self) -> float:
        """Safely get the current Z-coordinate."""
        with self.lock:
            return self.z_coord

    def stop(self) -> None:
        """Signal the application to stop."""
        with self.lock:
            self.running = False

    def is_running(self) -> bool:
        """Check if the application should be running."""
        with self.lock:
            return self.running


# --- Keyboard Listener ---
def keyboard_listener_thread(state: ControlState) -> None:
    """Listen for keyboard input and update the shared state."""

    def on_press(key: Any) -> bool:
        """Handle a key press event."""
        # Stop listener if the main app has stopped
        if not state.is_running():
            return False

        if key == keyboard.Key.up:
            state.adjust_z(Z_INCREMENT)
        elif key == keyboard.Key.down:
            state.adjust_z(-Z_INCREMENT)
        elif key == keyboard.Key.esc or (
            hasattr(key, "char") and key.char.lower() == "q"
        ):
            state.stop()
            return False  # Stop the listener
        return True

    # Start listening for key presses
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
    print("\nKeyboard listener stopped.")


# --- Main Application Logic ---
def main() -> None:
    """Run the main application loop for the Z-coordinate tester."""
    state = ControlState(initial_z=Z_START)

    # Start the keyboard listener in a separate thread so it doesn't block the main loop
    listener = threading.Thread(
        target=keyboard_listener_thread, args=(state,), daemon=True
    )
    listener.start()

    print("Connecting to Reachy Mini...")

    try:
        # The 'with' statement ensures the robot is properly put to sleep on exit
        with ReachyMini() as mini:
            print("Robot connected. Waking up...")
            mini.wake_up()

            print("\n" + "=" * 50)
            print("  Z-Coordinate Manual Controller")
            print(
                f"  Initial Z: {Z_START:.3f} m | Increment: {Z_INCREMENT * 1000:.1f} mm"
            )
            print("-" * 50)
            print("  CONTROLS: [Up/Down] to move | [Q/Esc] to quit")
            print("=" * 50 + "\n")

            # Main control loop
            while state.is_running():
                current_z = state.get_z()

                # Define the target pose: x=0, y=0, z=variable
                # Orientation (roll, pitch, yaw) is kept at zero.
                target_position = np.array([0, 0, current_z])
                target_orientation = np.zeros(3)

                mini.set_target(
                    utils.create_head_pose(
                        *target_position, *target_orientation, degrees=False
                    ),
                )

                # Print the sent value to the console, overwriting the previous line
                sys.stdout.write(f"\rSending Z-coordinate: {current_z:.4f} m  ")
                sys.stdout.flush()

                # Wait for a short period to maintain the control rate
                time.sleep(CONTROL_LOOP_RATE)

    except KeyboardInterrupt:
        print("\nCtrl-C received. Shutting down...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        state.stop()  # Ensure the listener thread can exit
        print("\n\nApplication finished. Robot is going to sleep.")


if __name__ == "__main__":
    main()
