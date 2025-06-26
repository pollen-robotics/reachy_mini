import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


def main():
    with ReachyMini() as mini:
        for method in ["linear", "minjerk", "ease", "cartoon"]:
            print(f"Testing method: {method}")

            pose = create_head_pose(x=0, y=0, z=0, yaw=0)
            mini.goto_target(pose, duration=1.0, method=method)

            for _ in range(3):
                pose = create_head_pose(
                    x=0.0, y=0.03, z=0, roll=5, yaw=-10, degrees=True
                )
                mini.goto_target(
                    pose,
                    antennas=list(np.deg2rad([-20, 20])),
                    duration=1.0,
                    method=method,
                )

                pose = create_head_pose(
                    x=0.0, y=-0.03, z=0, roll=-5, yaw=10, degrees=True
                )
                mini.goto_target(
                    pose,
                    antennas=list(np.deg2rad([20, -20])),
                    duration=1.0,
                    method=method,
                )

            pose = create_head_pose(x=0, y=0, z=0, yaw=0)
            mini.goto_target(pose, duration=1.0, antennas=[0, 0], method=method)


if __name__ == "__main__":
    main()
