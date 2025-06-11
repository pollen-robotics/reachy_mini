import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini


def main():
    with ReachyMini() as mini:
        t0 = time.time()
        while True:
            t = time.time() - t0
            target = np.deg2rad(30) * np.sin(2 * np.pi * 0.5 * t)

            yaw = target
            head = np.eye(4)
            head[:3, 3] = [0, 0, 0.177]
            head[:3, :3] = R.from_euler("xyz", [0, 0, yaw], degrees=False).as_matrix()

            mini.set_position(head=head, antennas=np.array([target, -target]))

            time.sleep(0.01)


if __name__ == "__main__":
    main()
