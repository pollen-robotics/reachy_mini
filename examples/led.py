import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini

with ReachyMini() as reachy_mini:
    try:
        while True:

            # Example 1: Set specific LEDs using dict
            print("Setting LEDs 0, 4, 8 to different colors...")
            reachy_mini.set_led_colors(
                {0: (255, 0, 0), 4: (0, 255, 0), 8: (0, 0, 255)}  # Red  # Green  # Blue
            )
            time.sleep(2)

            # Example 2: Set using list (None = don't change)
            print("Adding more colors, keeping previous ones...")
            colors = [None] * 12  # Start with all None
            colors[2] = (255, 255, 0)  # Yellow
            colors[6] = (255, 0, 255)  # Magenta
            colors[10] = (0, 255, 255)  # Cyan

            reachy_mini.set_led_colors(colors)
            time.sleep(2)

            # Example 4: Clear all
            print("Clearing all LEDs...")
            reachy_mini.clear_led()

            time.sleep(1)

    except KeyboardInterrupt:
        pass
