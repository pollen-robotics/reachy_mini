"""Reachy Mini Rerun viewer example.

Connects to a Reachy Mini, enables gravity compensation, and opens a Rerun
viewer that shows the live 3D URDF model with joint positions and camera feed.

Requires the 'rerun' extra: pip install reachy_mini[rerun]
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
