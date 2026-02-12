"""Reachy Mini Motion Sequence Example.

Demonstrates a variety of head motions using set_target() in a real-time loop.
Each step showcases a different degree of freedom or combination:
  1. Yaw oscillation (left-right)
  2. Pitch oscillation (up-down)
  3. Roll oscillation (tilt)
  4. Vertical translation (Z bounce)
  5. Antenna wave (alternating left/right)
  6. Circular XY translation
  7. Static pose sequence (position + yaw snaps)

The whole sequence loops until interrupted with Ctrl+C.
"""

# START doc_example

import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini


def main() -> None:
    """Run the motion sequence in a loop."""
    with ReachyMini(media_backend="no_media") as reachy_mini:
        reachy_mini.goto_target(np.eye(4), antennas=[0.0, 0.0], duration=1.0)
        try:
            while True:
                pose = np.eye(4)

                t = 0.0
                t0 = time.time()

                # --- Step 1: Yaw oscillation (0.7 rad amplitude, 0.5 Hz) ---
                s = time.time()
                while time.time() - s < 2.0:
                    t = time.time() - t0
                    euler_rot = np.array([0, 0.0, 0.7 * np.sin(2 * np.pi * 0.5 * t)])
                    rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
                    pose[:3, :3] = rot_mat
                    reachy_mini.set_target(head=pose, antennas=[0, 0])
                    time.sleep(0.01)

                # --- Step 2: Pitch oscillation (0.3 rad amplitude, 0.5 Hz) ---
                s = time.time()
                while time.time() - s < 2.0:
                    t = time.time() - t0
                    euler_rot = np.array([0, 0.3 * np.sin(2 * np.pi * 0.5 * t), 0])
                    rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
                    pose[:3, :3] = rot_mat
                    reachy_mini.set_target(head=pose, antennas=[0, 0])
                    time.sleep(0.01)

                # --- Step 3: Roll oscillation (0.3 rad amplitude, 0.5 Hz) ---
                s = time.time()
                while time.time() - s < 2.0:
                    t = time.time() - t0
                    euler_rot = np.array([0.3 * np.sin(2 * np.pi * 0.5 * t), 0, 0])
                    rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
                    pose[:3, :3] = rot_mat
                    reachy_mini.set_target(head=pose, antennas=[0, 0])
                    time.sleep(0.01)

                # --- Step 4: Vertical bounce (25 mm amplitude, 0.5 Hz) ---
                s = time.time()
                while time.time() - s < 2.0:
                    t = time.time() - t0
                    pose = np.eye(4)
                    pose[:3, 3][2] += 0.025 * np.sin(2 * np.pi * 0.5 * t)
                    reachy_mini.set_target(head=pose, antennas=[0, 0])
                    time.sleep(0.01)

                # --- Step 5: Antenna wave (alternating, 0.5 rad amplitude) ---
                s = time.time()
                while time.time() - s < 2.0:
                    t = time.time() - t0
                    antennas = [
                        0.5 * np.sin(2 * np.pi * 0.5 * t),
                        -0.5 * np.sin(2 * np.pi * 0.5 * t),
                    ]
                    reachy_mini.set_target(head=pose, antennas=antennas)
                    time.sleep(0.01)

                # --- Step 6: Circular XY translation (15 mm radius, 1 Hz) ---
                s = time.time()
                while time.time() - s < 5.0:
                    t = time.time() - t0
                    pose[:3, 3] = [
                        0.015 * np.sin(2 * np.pi * 1.0 * t),
                        0.015 * np.sin(2 * np.pi * 1.0 * t + np.pi / 2),
                        0.0,
                    ]
                    reachy_mini.set_target(head=pose, antennas=[0, 0])
                    time.sleep(0.01)

                # --- Step 7: Static pose sequence (position + yaw snaps) ---
                pose[:3, 3] = [0, 0, 0.0]
                reachy_mini.set_target(head=pose, antennas=[0, 0])

                time.sleep(0.5)

                pose[:3, 3] = [0.02, 0.02, 0.0]
                reachy_mini.set_target(head=pose, antennas=[0, 0])
                time.sleep(0.5)

                pose[:3, 3] = [0.00, 0.02, 0.0]
                euler_rot = np.array([0, 0, 0.5])
                rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
                pose[:3, :3] = rot_mat
                reachy_mini.set_target(head=pose, antennas=[0, 0])
                time.sleep(0.5)

                pose[:3, 3] = [0.00, -0.02, 0.0]
                euler_rot = np.array([0, 0, -0.5])
                rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
                pose[:3, :3] = rot_mat
                reachy_mini.set_target(head=pose, antennas=[0, 0])
                time.sleep(0.5)

                pose[:3, 3] = [0, 0, 0.0]
                reachy_mini.set_target(head=pose, antennas=[0, 0])
                time.sleep(2)

        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

# END doc_example
