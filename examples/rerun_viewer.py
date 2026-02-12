"""Reachy Mini Rerun Viewer Example.

This example shows how to use the Rerun utility to log and visualize Reachy Mini's state.
It is based on the gravity compensation example, so the robot will be compliant and easy to move around.

Requirements:
- Install with: pip install reachy-mini[rerun,placo_kinematics]
- Start the daemon with: reachy-mini-daemon --kinematics-engine Placo
"""

# START doc_example

import logging
import time

from reachy_mini import ReachyMini
from reachy_mini.utils.rerun import Rerun


def main() -> None:
    """Log and visualize Reachy Mini's state using Rerun."""
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    with ReachyMini(log_level="DEBUG") as mini:
        try:
            mini.enable_gravity_compensation()
            rerun = Rerun(mini)
            rerun.start()

            print("Reachy Mini is now compliant. Press Ctrl+C to exit.")
            while True:
                # do nothing, just keep the program running
                time.sleep(0.02)

        except KeyboardInterrupt:
            pass
        finally:
            mini.disable_gravity_compensation()
            rerun.stop()
            print("Exiting... Reachy Mini is stiff again.")


if __name__ == "__main__":
    main()

# END doc_example
