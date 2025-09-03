"""Reachy Mini Compliant Demo.

This demo turns the Reachy Mini into compliant mode and compensates for the gravity of the robot platform to prevent it from falling down.

You can now gently push the robot and it will follow your movements. And when you stop pushing it, it will stay in place.
This is useful for applications like human-robot interaction, where you want the robot to be compliant and follow the user's movements.
"""

import time

from reachy_mini import ReachyMini

with ReachyMini() as mini:
    try:
        mini.enable_gravity_compensation()

        print("Reachy Mini is now compliant. Press Ctrl+C to exit.")
        while True:
            # do nothing, just keep the program running
            time.sleep(0.02)

    except KeyboardInterrupt:
        mini.disable_gravity_compensation()
        print("Exiting... Reachy Mini is stiff again.")
