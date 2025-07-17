#!/usr/bin/env python3
"""Control Reachy Mini's head yaw angle with a joystick.

This script connects to a Reachy Mini robot and allows you to pilot its head's
left-right rotation (yaw) using the horizontal axis of a connected joystick.

The yaw angle is mapped to a full range of +-pi/2 radians (+-90 degrees).
The value from the right joystick is also printed but is not used for control.

CONTROLS:
- LEFT JOYSTICK (Left/Right): Control head yaw angle.
- CIRCLE / B BUTTON (Button 1): Quit the application safely.
- CTRL-C: Quit the application.
"""

# Standard library imports
import os
import sys
import time

# Third-party imports
import numpy as np
import pygame

# Local application/library-specific imports
from reachy_mini import ReachyMini, utils

# --- Configuration ---
CONTROL_LOOP_RATE = 0.02
# Maximum yaw angle. The joystick's -1 to 1 input will be mapped to this range.
YAW_ANGLE_LIMIT = np.pi / 4 * 1.3  # Radians

# To use pygame "headlessly" (without a GUI window).
os.environ["SDL_VIDEODRIVER"] = "dummy"

# --- Controller Bindings Comment ---
# PS4 controller:
# Button 1 = O (Circle)
# Axis 0: Left Joy Left/Right   (-1 left, 1 right)
# Axis 3 or 4: Right Joy Left/Right
#
# XBOX controller:
# Button 1 = B
# Axis 0: Left Joy Left/Right
# Axis 2 or 3: Right Joy Left/Right


class Controller:
    """Handle joystick input using pygame."""

    def __init__(self, deadzone: float = 0.08):
        """Initialize the controller and find the first joystick.

        Args:
            deadzone (float): Axis value below which input is ignored.

        Raises:
            IOError: If no joystick is found.

        """
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() < 1:
            raise IOError("No joystick controller found.")

        self.joystick: pygame.joystick.Joystick = pygame.joystick.Joystick(0)
        self.deadzone = deadzone
        print(f"Initialized joystick: {self.joystick.get_name()}")

    def _apply_deadzone(self, value: float) -> float:
        """Apply a deadzone to a joystick axis value."""
        return value if abs(value) > self.deadzone else 0.0

    def get_horizontal_inputs(self) -> tuple[float, float]:
        """Read the horizontal axes of the left and right joysticks.

        Returns:
            tuple[float, float]: (left_joy_h, right_joy_h) from -1.0 to 1.0.

        """
        pygame.event.pump()  # Update pygame's internal event state.

        left_joy_h = self._apply_deadzone(self.joystick.get_axis(0))

        # Right joystick horizontal axis can be 2, 3 or 4 depending on controller
        right_joy_h = 0.0
        if self.joystick.get_numaxes() > 3:
            right_joy_h = self._apply_deadzone(self.joystick.get_axis(3))
        elif self.joystick.get_numaxes() > 2:
            right_joy_h = self._apply_deadzone(self.joystick.get_axis(2))

        return left_joy_h, right_joy_h

    def check_for_quit(self) -> bool:
        """Check pygame events for a quit signal.

        Returns:
            bool: True if the designated quit button (Circle/B) is pressed.

        """
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if self.joystick.get_button(1):  # Button 1 is Circle/B
                    print("\nQuit button pressed.")
                    return True
        return False


def main() -> None:
    """Run the main joystick control loop."""
    try:
        controller = Controller()
    except IOError as e:
        print(f"Error: {e}", file=sys.stderr)
        return

    print("Connecting to Reachy Mini...")
    try:
        # The 'with' statement ensures the robot is properly handled on exit
        with ReachyMini(automatic_body_yaw=True) as mini:
            print("Robot connected.")
            # print("Robot connected. Waking up...")
            # mini.wake_up()

            print("\n" + "=" * 50)
            print("  Reachy Head Yaw Joystick Controller")
            print("  CONTROLS: [Left Stick] to turn | [Circle/B] to quit")
            print("=" * 50 + "\n")

            while True:
                if controller.check_for_quit():
                    break

                # Get scaled joystick values
                left_joy, right_joy = controller.get_horizontal_inputs()

                # Map joystick input (-1 to 1) to the desired angle range
                target_yaw = left_joy * YAW_ANGLE_LIMIT

                target_body_yaw = right_joy * YAW_ANGLE_LIMIT

                # Define the target pose: x,y,z and roll,pitch,yaw
                target_position = np.array([0, 0, 0.0])
                target_orientation = np.array([0, 0, target_yaw])

                # Create and send the command to the robot
                mini.set_target(
                    utils.create_head_pose(
                        *target_position, *target_orientation, degrees=False
                    ),
                    body_yaw=target_body_yaw,
                )

                # Print status, overwriting the line
                print(
                    f"\rSending Yaw: {target_yaw:6.2f} rad | "
                    f"Unused Right Joy: {right_joy:6.2f}",
                    end="",
                )
                sys.stdout.flush()

                time.sleep(CONTROL_LOOP_RATE)

    except KeyboardInterrupt:
        print("\nCTRL+C detected. Shutting down...")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
    finally:
        print("\n\nApplication finished. Robot will go to sleep.")
        pygame.quit()


if __name__ == "__main__":
    main()
