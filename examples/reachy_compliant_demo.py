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
        # set torque control mode
        mini.make_compliant(head=True, antennas=True)

        print("Reachy Mini is now compliant. Press Ctrl+C to exit.")
        while True:
            # compensate the gravity of the robot platform
            # this is useful to avoid the robot to fall down when it is compliant
            # > it is optional, but it is recommended to use it so that the robot does not fall down
            mini.compensate_gravity()

            time.sleep(0.02)
    except KeyboardInterrupt:
        mini.make_compliant(head=False, antennas=False)
        print("Exiting... Reachy Mini is stiff again.")
