"""Reachy Mini sound playback example.

Open a wav and push samples to the speaker. This is a toy example, in real
conditions output from a microphone or a text-to-speech engine would be
 pushed to the speaker instead.
"""

import logging
import time

from reachy_mini import ReachyMini


def main():
    """Play a wav file by pushing samples to the audio device."""
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    with ReachyMini(log_level="DEBUG") as mini:
        while True:
            doa = mini.media.audio.get_DoA()
            print(f"DOA: {doa}")
            if doa[1]:
                print(f"  Speech detected at {doa[0]:.1f}°")
                x = 1.0
                z = 0.0
                y = 1 - doa[0] / 90.0
                print(f"  Pointing to x={x:.2f}, y={y:.2f}, z={z:.2f}")
                mini.look_at_world(x, y, z, duration=0.5)
            else:
                time.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
