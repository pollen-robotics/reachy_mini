"""Reachy Mini sound playback example.

Open a wav and push samples to the speaker. This is a toy example, in real
conditions output from a microphone or a text-to-speech engine would be
 pushed to the speaker instead.
"""

import logging
import time

import numpy as np

from reachy_mini import ReachyMini


def main() -> None:
    """Play a wav file by pushing samples to the audio device."""
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    with ReachyMini(log_level="DEBUG", automatic_body_yaw=True) as mini:
        last_doa = -1
        THRESHOLD = 0.004  # ~2 degrees
        while True:
            doa = mini.media.audio.get_DoA()
            print(f"DOA: {doa}")
            if doa[1] and np.abs(doa[0] - last_doa) > THRESHOLD:
                print(f"  Speech detected at {doa[0]:.1f}°")
                p_head = [np.sin(doa[0]), np.cos(doa[0]), 0.0]
                print(
                    f"  Pointing to x={p_head[0]:.2f}, y={p_head[1]:.2f}, z={p_head[2]:.2f}"
                )
                T_world_head = mini.get_current_head_pose()
                R_world_head = T_world_head[:3, :3]
                p_world = R_world_head @ p_head
                print(
                    f"  In world coordinates: x={p_world[0]:.2f}, y={p_world[1]:.2f}, z={p_world[2]:.2f}"
                )
                mini.look_at_world(*p_world, duration=0.5)
                last_doa = doa[0]
            else:
                if not doa[1]:
                    print("  No speech detected")
                else:
                    print(
                        f"  Small change in DOA: {doa[0]:.1f}° (last was {last_doa:.1f}°). Not moving."
                    )
                time.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
