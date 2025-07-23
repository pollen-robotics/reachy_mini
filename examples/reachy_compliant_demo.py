"""Reachy Mini Compliant Demo.

First, this demo turns the Reachy Mini into compliant mode.
Then, it compensates for the gravity of the robot platform to prevent it from falling down.

You can now gently push the robot and it will follow your movements. And when you stop pushing it, it will stay in place.
This is useful for applications like human-robot interaction, where you want the robot to be compliant and follow the user's movements.
"""

import time

from reachy_mini import ReachyMini

with ReachyMini() as mini:
    try:
        # Set the head and antennas to compliant mode
        # With compensate_gravity=True, the robot will compensate for gravity
        # and will not fall down when you push it.
        # If you want to disable gravity compensation, set compensate_gravity=False.
        mini.make_motors_compliant(head=True, antennas=True, compensate_gravity=True)

        print("Reachy Mini is now compliant. Press Ctrl+C to exit.")
        while True:
            # do nothing, just keep the program running
            time.sleep(0.02)

    except KeyboardInterrupt:
        mini.make_motors_compliant(head=False, antennas=False)
        print("Exiting... Reachy Mini is stiff again.")
