"""Reachy Mini Compliant Demo."""

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
