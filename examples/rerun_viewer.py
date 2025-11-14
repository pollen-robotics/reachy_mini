"""Reachy Mini sound playback example.

Open a wav and push samples to the speaker. This is a toy example, in real
conditions output from a microphone or a text-to-speech engine would be
 pushed to the speaker instead.

It requires the 'rerun-loader-urdf' package to be installed. It's not on PyPI,
so you need to install it from the GitHub repository: pip install git+https://github.com/rerun-io/rerun-loader-python-example-urdf.git
"""

import logging
import time

from reachy_mini import ReachyMini
from reachy_mini.utils.rerun import Rerun


def main():
    """Play a wav file by pushing samples to the audio device."""
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
            mini.disable_gravity_compensation()
            rerun.stop()
            print("Exiting... Reachy Mini is stiff again.")


if __name__ == "__main__":
    main()
