import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini

with ReachyMini("/dev/ttyUSB0") as reachy_mini:
    try:
        while True:

            ring = reachy_mini.led  # Linux

            # Example 1: Set specific LEDs using dict
            print("Setting LEDs 0, 4, 8 to different colors...")
            reachy_mini.led.set_colors(
                {0: (255, 0, 0), 4: (0, 255, 0), 8: (0, 0, 255)}  # Red  # Green  # Blue
            )
            time.sleep(2)

            # Example 2: Set using list (None = don't change)
            print("Adding more colors, keeping previous ones...")
            colors = [None] * 12  # Start with all None
            colors[2] = (255, 255, 0)  # Yellow
            colors[6] = (255, 0, 255)  # Magenta
            colors[10] = (0, 255, 255)  # Cyan

            reachy_mini.led.set_colors(colors)
            time.sleep(2)

            # Example 3: Get current status
            print("Current LED states:")
            status = reachy_mini.led.get_status()
            for led_id, (r, g, b) in status.items():
                if r > 0 or g > 0 or b > 0:  # Only show active LEDs
                    print(f"  LED {led_id}: RGB({r}, {g}, {b})")

            # Example 4: Clear all
            print("Clearing all LEDs...")
            reachy_mini.led.clear()

            time.sleep(1)

    except KeyboardInterrupt:
        pass
