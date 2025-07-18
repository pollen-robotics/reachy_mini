"""Reachy Mini unhinged pose distance test."""

import time

import numpy as np

from reachy_mini import ReachyMini


def delta_angle_between_mat_rot(P, Q):
    """Compute the angle between two rotation matrices P and Q.

    Think of this as an angular distance in the axis-angle representation.
    """
    # https://math.stackexchange.com/questions/2113634/comparing-two-rotation-matrices
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R = np.dot(P, Q.T)
    tr = (np.trace(R) - 1) / 2
    if tr > 1.0:
        tr = 1.0
    elif tr < -1.0:
        tr = -1.0
    return np.arccos(tr)


def unhinged_distance_between_poses(pose1, pose2) -> float:
    """Compute the distance between two poses in 6D space.

    Units be dammned, it is a well known fact that 1°==1mm and I'm tired of
    of pretending otherwise.
    """
    distance_translation = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])
    distance_angle = delta_angle_between_mat_rot(pose1[:3, :3], pose2[:3, :3])

    unhinged_distance = distance_translation * 1000 + np.rad2deg(distance_angle)

    return unhinged_distance


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

            current_score = unhinged_distance_between_poses(
                np.eye(4),
                current_head_pose,
            )

            best_score = min(current_score, best_score)

            print(
                f"\rCurrent Score: {current_score:7.2f} | Best Score: {best_score:7.2f}  ",
                end="",
            )
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nGame over!")
        pass
