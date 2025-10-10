"""Reachy Mini Goto Target Interpolation Playground.

This example demonstrates the different interpolation methods available in Reachy Mini
for moving the head and/or antennas to a target pose. It tests various methods such as linear,
minjerk, ease, and cartoon, allowing the user to observe how each method affects the
motion of the head and antennas.
"""

import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import InterpolationTechnique


def main():
    """Run the different interpolation methods."""
    with ReachyMini(media_backend="no_media") as mini:
        try:
            for method in InterpolationTechnique:
                print(f"Testing method: {method}")

                pose = create_head_pose(x=0, y=0, z=0, yaw=0)
                mini.goto_target(pose, duration=1.0, method=method)

                for _ in range(3):
                    pose = create_head_pose(
                        x=0.0, y=0.03, z=0, roll=5, yaw=-10, degrees=True
                    )
                    mini.goto_target(
                        pose,
                        antennas=np.deg2rad([-20, 20]),
                        duration=1.0,
                        method=method,
                    )

                    pose = create_head_pose(
                        x=0.0, y=-0.03, z=0, roll=-5, yaw=10, degrees=True
                    )
                    mini.goto_target(
                        pose,
                        antennas=np.deg2rad([20, -20]),
                        duration=1.0,
                        method=method,
                    )

                pose = create_head_pose(x=0, y=0, z=0, yaw=0)
                mini.goto_target(pose, duration=1.0, antennas=[0, 0], method=method)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
