"""Reachy Mini sound Direction of Arrival (DoA) example.

This script demonstrates how to use the microphone array to detect the
Direction of Arrival (DoA) of speech using the FastAPI endpoint. It calculates
the position of the sound source relative to the head, transforms it into world
coordinates, and commands the robot to look towards the speaker.
"""

# START doc_example

import time

import numpy as np
import requests

from reachy_mini import ReachyMini


def main() -> None:
    """Continuously monitor audio input and orient the head toward the speaker."""
    with ReachyMini(log_level="DEBUG", automatic_body_yaw=True) as mini:
        # Get the robot IP address
        robot_ip = "localhost"
        status = mini.client.get_status()
        if status.wireless_version and status.wlan_ip:
            robot_ip = status.wlan_ip

        doa_url = f"http://{robot_ip}:8000/api/state/doa"

        last_doa = -1
        THRESHOLD = 0.004  # ~2 degrees

        while True:
            # Get DoA from FastAPI endpoint
            try:
                response = requests.get(doa_url, timeout=1.0)
                response.raise_for_status()
                doa_data = response.json()

                if doa_data is None:
                    print("  No DoA data available (audio not initialized)")
                    time.sleep(0.5)
                    continue

                angle = doa_data["angle"]
                speech_detected = doa_data["speech_detected"]

                print(f"DOA: angle={angle:.3f} rad, speech_detected={speech_detected}")

                if speech_detected and np.abs(angle - last_doa) > THRESHOLD:
                    print(f"  Speech detected at {angle:.1f} radians")
                    p_head = [np.sin(angle), np.cos(angle), 0.0]
                    print(
                        f"  Pointing to x={p_head[0]:.2f}, y={p_head[1]:.2f}, z={p_head[2]:.2f}"
                    )
                    T_world_head = mini.get_current_head_pose()
                    R_world_head = T_world_head[:3, :3]
                    p_world = R_world_head @ p_head
                    print(
                        f"  In world coordinates: x={p_world[0]:.2f}, y={p_world[1]:.2f}, z={p_world[2]:.2f}"
                    )
                    mini.look_at_world(p_world[0], p_world[1], p_world[2], duration=0.5)
                    last_doa = angle
                else:
                    if not speech_detected:
                        print("  No speech detected")
                    else:
                        print(
                            f"  Small change in DOA: {angle:.1f} rad (last was {last_doa:.1f} rad). Not moving."
                        )
                    time.sleep(0.5)

            except requests.RequestException as e:
                print(f"  Error fetching DoA: {e}")
                time.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")

# END doc_example
