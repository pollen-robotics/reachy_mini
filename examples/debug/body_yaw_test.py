"""Reachy Mini Head Position GUI Example."""

import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini


def main():
    """Run a GUI to set the head position and orientation of Reachy Mini."""
    with ReachyMini() as mini:
        # with ReachyMini(automatic_body_yaw=False) as mini:
        t0 = time.time()

        while True:
            t = time.time() - t0
            target = np.deg2rad(90) * np.sin(2 * np.pi * 0.5 * t)

            head = np.eye(4)
            head[:3, 3] = [0, 0, 0]

            # Read values from the GUI
            roll = np.deg2rad(0.0)
            pitch = np.deg2rad(0.0)
            yaw = np.deg2rad(0.0)
            head[:3, :3] = R.from_euler(
                "xyz", [roll, pitch, yaw], degrees=False
            ).as_matrix()

            mini.set_target(
                head=head,
                antennas=np.array([target, -target]),
                body_yaw=target,
            )
            time.sleep(0.01)


if __name__ == "__main__":
    main()
