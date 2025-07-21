"""Reachy Mini magical pose distance test."""

import time

import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import distance_between_poses

check_collision = False  # Set to True to enable collision checking
with ReachyMini() as reachy_mini:
    best_score = float("inf")
    try:
        reachy_mini.disable_motors()
        print("Try to get the lowest score! Press Ctrl+C to exit.")
        time.sleep(1)  # Give user time to read instructions

        while True:
            cur_head_joints, _ = reachy_mini._get_current_joint_positions()
            current_head_pose = reachy_mini.head_kinematics.fk(cur_head_joints)

            _, _, magic_distance = distance_between_poses(
                np.eye(4),
                current_head_pose,
            )

            best_score = min(magic_distance, best_score)

            print(
                f"\rCurrent Score: {magic_distance:7.2f} | Best Score: {best_score:7.2f}  ",
                end="",
            )
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nGame over!")
        pass
