from math import atan, sqrt
import numpy as np
import time
import placo
from placo_utils.visualization import robot_viz, robot_frame_viz, frame_viz
from placo_utils.tf import tf
from reachy_mini.placo_kinematics import PlacoKinematics

import pinocchio as pin
import tkinter as tk

urdf_path = "/home/gospar/pollen_robotics/reachy_mini/src/reachy_mini/descriptions/reachy_mini/urdf/robot.urdf"
solver = PlacoKinematics(urdf_path, 0.02)
robot = solver.robot
robot.update_kinematics()

viz = robot_viz(robot)

motors = [
    {"name": "1", "branch_frame": "closing_1_2", "offset": 0, "solution": 0},
    {"name": "2", "branch_frame": "closing_2_2", "offset": 0, "solution": 1},
    {"name": "3", "branch_frame": "closing_3_2", "offset": 0, "solution": 0},
    {"name": "4", "branch_frame": "closing_4_2", "offset": 0, "solution": 1},
    {"name": "5", "branch_frame": "closing_5_2", "offset": 0, "solution": 0},
    {"name": "6", "branch_frame": "passive_7_link_y", "offset": 0, "solution": 1},
]

# Measuring lengths for the arm and branch (constants could be used)
T_world_head = robot.get_T_world_frame("head").copy()
T_world_1 = robot.get_T_world_frame("1")
T_world_arm1 = robot.get_T_world_frame("passive_1_link_x")

T_1_arm1 = np.linalg.inv(T_world_1) @ T_world_arm1
arm_z = T_1_arm1[2, 3]
servo_arm_length = np.linalg.norm(T_1_arm1[:2, 3])
T_world_branch1 = robot.get_T_world_frame("closing_1_2")
T_arm1_branch1 = np.linalg.inv(T_world_arm1) @ T_world_branch1
branch_length = np.linalg.norm(T_arm1_branch1[:3, 3])

# Finding the 6 branches position in the head frame, and building T_world_motor frame
# T_world_motor is a frame where the motor is located, correcting for the arm_z offset
for motor in motors:
    T_world_branch = robot.get_T_world_frame(motor["branch_frame"])
    T_head_branch = np.linalg.inv(T_world_head) @ T_world_branch
    T_world_motor = robot.get_T_world_frame(motor["name"]) @ tf.translation_matrix(
        (0, 0, arm_z)
    )
    motor["T_world_motor"] = T_world_motor
    motor["branch_position"] = T_head_branch[:3, 3]


np.set_printoptions(precision=2, suppress=True)


# See compute_analytical_kinematics.py
def ik_branch(px, py, pz):
    rs = servo_arm_length
    rp = branch_length

    return [
        2
        * atan(
            (
                2 * py * rs
                - sqrt(
                    -(px**4)
                    - 2 * px**2 * py**2
                    - 2 * px**2 * pz**2
                    + 2 * px**2 * rp**2
                    + 2 * px**2 * rs**2
                    - py**4
                    - 2 * py**2 * pz**2
                    + 2 * py**2 * rp**2
                    + 2 * py**2 * rs**2
                    - pz**4
                    + 2 * pz**2 * rp**2
                    - 2 * pz**2 * rs**2
                    - rp**4
                    + 2 * rp**2 * rs**2
                    - rs**4
                )
            )
            / (px**2 + 2 * px * rs + py**2 + pz**2 - rp**2 + rs**2)
        ),
        2
        * atan(
            (
                2 * py * rs
                + sqrt(
                    -(px**4)
                    - 2 * px**2 * py**2
                    - 2 * px**2 * pz**2
                    + 2 * px**2 * rp**2
                    + 2 * px**2 * rs**2
                    - py**4
                    - 2 * py**2 * pz**2
                    + 2 * py**2 * rp**2
                    + 2 * py**2 * rs**2
                    - pz**4
                    + 2 * pz**2 * rp**2
                    - 2 * pz**2 * rs**2
                    - rp**4
                    + 2 * rp**2 * rs**2
                    - rs**4
                )
            )
            / (px**2 + 2 * px * rs + py**2 + pz**2 - rp**2 + rs**2)
        ),
    ]


def jacobian_branch(cx, cy, cz):
    ml = servo_arm_length
    bl = branch_length

    return np.array(
        [
            [
                4
                * (
                    cx
                    * (-(bl**2) + cx**2 + cy**2 + cz**2 - ml**2)
                    * (-(bl**2) + cx**2 + 2 * cx * ml + cy**2 + cz**2 + ml**2)
                    - (cx + ml)
                    * (
                        2 * cy * ml
                        - sqrt(
                            -(bl**4)
                            + 2 * bl**2 * cx**2
                            + 2 * bl**2 * cy**2
                            + 2 * bl**2 * cz**2
                            + 2 * bl**2 * ml**2
                            - cx**4
                            - 2 * cx**2 * cy**2
                            - 2 * cx**2 * cz**2
                            + 2 * cx**2 * ml**2
                            - cy**4
                            - 2 * cy**2 * cz**2
                            + 2 * cy**2 * ml**2
                            - cz**4
                            - 2 * cz**2 * ml**2
                            - ml**4
                        )
                    )
                    * sqrt(
                        -(bl**4)
                        + 2 * bl**2 * cx**2
                        + 2 * bl**2 * cy**2
                        + 2 * bl**2 * cz**2
                        + 2 * bl**2 * ml**2
                        - cx**4
                        - 2 * cx**2 * cy**2
                        - 2 * cx**2 * cz**2
                        + 2 * cx**2 * ml**2
                        - cy**4
                        - 2 * cy**2 * cz**2
                        + 2 * cy**2 * ml**2
                        - cz**4
                        - 2 * cz**2 * ml**2
                        - ml**4
                    )
                )
                / (
                    (
                        (
                            2 * cy * ml
                            - sqrt(
                                -(bl**4)
                                + 2 * bl**2 * cx**2
                                + 2 * bl**2 * cy**2
                                + 2 * bl**2 * cz**2
                                + 2 * bl**2 * ml**2
                                - cx**4
                                - 2 * cx**2 * cy**2
                                - 2 * cx**2 * cz**2
                                + 2 * cx**2 * ml**2
                                - cy**4
                                - 2 * cy**2 * cz**2
                                + 2 * cy**2 * ml**2
                                - cz**4
                                - 2 * cz**2 * ml**2
                                - ml**4
                            )
                        )
                        ** 2
                        + (-(bl**2) + cx**2 + 2 * cx * ml + cy**2 + cz**2 + ml**2) ** 2
                    )
                    * sqrt(
                        -(bl**4)
                        + 2 * bl**2 * cx**2
                        + 2 * bl**2 * cy**2
                        + 2 * bl**2 * cz**2
                        + 2 * bl**2 * ml**2
                        - cx**4
                        - 2 * cx**2 * cy**2
                        - 2 * cx**2 * cz**2
                        + 2 * cx**2 * ml**2
                        - cy**4
                        - 2 * cy**2 * cz**2
                        + 2 * cy**2 * ml**2
                        - cz**4
                        - 2 * cz**2 * ml**2
                        - ml**4
                    )
                ),
                4
                * (
                    -cy
                    * (
                        2 * cy * ml
                        - sqrt(
                            -(bl**4)
                            + 2 * bl**2 * cx**2
                            + 2 * bl**2 * cy**2
                            + 2 * bl**2 * cz**2
                            + 2 * bl**2 * ml**2
                            - cx**4
                            - 2 * cx**2 * cy**2
                            - 2 * cx**2 * cz**2
                            + 2 * cx**2 * ml**2
                            - cy**4
                            - 2 * cy**2 * cz**2
                            + 2 * cy**2 * ml**2
                            - cz**4
                            - 2 * cz**2 * ml**2
                            - ml**4
                        )
                    )
                    * sqrt(
                        -(bl**4)
                        + 2 * bl**2 * cx**2
                        + 2 * bl**2 * cy**2
                        + 2 * bl**2 * cz**2
                        + 2 * bl**2 * ml**2
                        - cx**4
                        - 2 * cx**2 * cy**2
                        - 2 * cx**2 * cz**2
                        + 2 * cx**2 * ml**2
                        - cy**4
                        - 2 * cy**2 * cz**2
                        + 2 * cy**2 * ml**2
                        - cz**4
                        - 2 * cz**2 * ml**2
                        - ml**4
                    )
                    + (-(bl**2) + cx**2 + 2 * cx * ml + cy**2 + cz**2 + ml**2)
                    * (
                        -(bl**2) * cy
                        + cx**2 * cy
                        + cy**3
                        + cy * cz**2
                        - cy * ml**2
                        + ml
                        * sqrt(
                            -(bl**4)
                            + 2 * bl**2 * cx**2
                            + 2 * bl**2 * cy**2
                            + 2 * bl**2 * cz**2
                            + 2 * bl**2 * ml**2
                            - cx**4
                            - 2 * cx**2 * cy**2
                            - 2 * cx**2 * cz**2
                            + 2 * cx**2 * ml**2
                            - cy**4
                            - 2 * cy**2 * cz**2
                            + 2 * cy**2 * ml**2
                            - cz**4
                            - 2 * cz**2 * ml**2
                            - ml**4
                        )
                    )
                )
                / (
                    (
                        (
                            2 * cy * ml
                            - sqrt(
                                -(bl**4)
                                + 2 * bl**2 * cx**2
                                + 2 * bl**2 * cy**2
                                + 2 * bl**2 * cz**2
                                + 2 * bl**2 * ml**2
                                - cx**4
                                - 2 * cx**2 * cy**2
                                - 2 * cx**2 * cz**2
                                + 2 * cx**2 * ml**2
                                - cy**4
                                - 2 * cy**2 * cz**2
                                + 2 * cy**2 * ml**2
                                - cz**4
                                - 2 * cz**2 * ml**2
                                - ml**4
                            )
                        )
                        ** 2
                        + (-(bl**2) + cx**2 + 2 * cx * ml + cy**2 + cz**2 + ml**2) ** 2
                    )
                    * sqrt(
                        -(bl**4)
                        + 2 * bl**2 * cx**2
                        + 2 * bl**2 * cy**2
                        + 2 * bl**2 * cz**2
                        + 2 * bl**2 * ml**2
                        - cx**4
                        - 2 * cx**2 * cy**2
                        - 2 * cx**2 * cz**2
                        + 2 * cx**2 * ml**2
                        - cy**4
                        - 2 * cy**2 * cz**2
                        + 2 * cy**2 * ml**2
                        - cz**4
                        - 2 * cz**2 * ml**2
                        - ml**4
                    )
                ),
                4
                * cz
                * (
                    -(
                        2 * cy * ml
                        - sqrt(
                            -(bl**4)
                            + 2 * bl**2 * cx**2
                            + 2 * bl**2 * cy**2
                            + 2 * bl**2 * cz**2
                            + 2 * bl**2 * ml**2
                            - cx**4
                            - 2 * cx**2 * cy**2
                            - 2 * cx**2 * cz**2
                            + 2 * cx**2 * ml**2
                            - cy**4
                            - 2 * cy**2 * cz**2
                            + 2 * cy**2 * ml**2
                            - cz**4
                            - 2 * cz**2 * ml**2
                            - ml**4
                        )
                    )
                    * sqrt(
                        -(bl**4)
                        + 2 * bl**2 * cx**2
                        + 2 * bl**2 * cy**2
                        + 2 * bl**2 * cz**2
                        + 2 * bl**2 * ml**2
                        - cx**4
                        - 2 * cx**2 * cy**2
                        - 2 * cx**2 * cz**2
                        + 2 * cx**2 * ml**2
                        - cy**4
                        - 2 * cy**2 * cz**2
                        + 2 * cy**2 * ml**2
                        - cz**4
                        - 2 * cz**2 * ml**2
                        - ml**4
                    )
                    + (-(bl**2) + cx**2 + cy**2 + cz**2 + ml**2)
                    * (-(bl**2) + cx**2 + 2 * cx * ml + cy**2 + cz**2 + ml**2)
                )
                / (
                    (
                        (
                            2 * cy * ml
                            - sqrt(
                                -(bl**4)
                                + 2 * bl**2 * cx**2
                                + 2 * bl**2 * cy**2
                                + 2 * bl**2 * cz**2
                                + 2 * bl**2 * ml**2
                                - cx**4
                                - 2 * cx**2 * cy**2
                                - 2 * cx**2 * cz**2
                                + 2 * cx**2 * ml**2
                                - cy**4
                                - 2 * cy**2 * cz**2
                                + 2 * cy**2 * ml**2
                                - cz**4
                                - 2 * cz**2 * ml**2
                                - ml**4
                            )
                        )
                        ** 2
                        + (-(bl**2) + cx**2 + 2 * cx * ml + cy**2 + cz**2 + ml**2) ** 2
                    )
                    * sqrt(
                        -(bl**4)
                        + 2 * bl**2 * cx**2
                        + 2 * bl**2 * cy**2
                        + 2 * bl**2 * cz**2
                        + 2 * bl**2 * ml**2
                        - cx**4
                        - 2 * cx**2 * cy**2
                        - 2 * cx**2 * cz**2
                        + 2 * cx**2 * ml**2
                        - cy**4
                        - 2 * cy**2 * cz**2
                        + 2 * cy**2 * ml**2
                        - cz**4
                        - 2 * cz**2 * ml**2
                        - ml**4
                    )
                ),
            ],
            [
                4
                * (
                    cx
                    * (bl**2 - cx**2 - cy**2 - cz**2 + ml**2)
                    * (-(bl**2) + cx**2 + 2 * cx * ml + cy**2 + cz**2 + ml**2)
                    - (cx + ml)
                    * (
                        2 * cy * ml
                        + sqrt(
                            -(bl**4)
                            + 2 * bl**2 * cx**2
                            + 2 * bl**2 * cy**2
                            + 2 * bl**2 * cz**2
                            + 2 * bl**2 * ml**2
                            - cx**4
                            - 2 * cx**2 * cy**2
                            - 2 * cx**2 * cz**2
                            + 2 * cx**2 * ml**2
                            - cy**4
                            - 2 * cy**2 * cz**2
                            + 2 * cy**2 * ml**2
                            - cz**4
                            - 2 * cz**2 * ml**2
                            - ml**4
                        )
                    )
                    * sqrt(
                        -(bl**4)
                        + 2 * bl**2 * cx**2
                        + 2 * bl**2 * cy**2
                        + 2 * bl**2 * cz**2
                        + 2 * bl**2 * ml**2
                        - cx**4
                        - 2 * cx**2 * cy**2
                        - 2 * cx**2 * cz**2
                        + 2 * cx**2 * ml**2
                        - cy**4
                        - 2 * cy**2 * cz**2
                        + 2 * cy**2 * ml**2
                        - cz**4
                        - 2 * cz**2 * ml**2
                        - ml**4
                    )
                )
                / (
                    (
                        (
                            2 * cy * ml
                            + sqrt(
                                -(bl**4)
                                + 2 * bl**2 * cx**2
                                + 2 * bl**2 * cy**2
                                + 2 * bl**2 * cz**2
                                + 2 * bl**2 * ml**2
                                - cx**4
                                - 2 * cx**2 * cy**2
                                - 2 * cx**2 * cz**2
                                + 2 * cx**2 * ml**2
                                - cy**4
                                - 2 * cy**2 * cz**2
                                + 2 * cy**2 * ml**2
                                - cz**4
                                - 2 * cz**2 * ml**2
                                - ml**4
                            )
                        )
                        ** 2
                        + (-(bl**2) + cx**2 + 2 * cx * ml + cy**2 + cz**2 + ml**2) ** 2
                    )
                    * sqrt(
                        -(bl**4)
                        + 2 * bl**2 * cx**2
                        + 2 * bl**2 * cy**2
                        + 2 * bl**2 * cz**2
                        + 2 * bl**2 * ml**2
                        - cx**4
                        - 2 * cx**2 * cy**2
                        - 2 * cx**2 * cz**2
                        + 2 * cx**2 * ml**2
                        - cy**4
                        - 2 * cy**2 * cz**2
                        + 2 * cy**2 * ml**2
                        - cz**4
                        - 2 * cz**2 * ml**2
                        - ml**4
                    )
                ),
                4
                * (
                    -cy
                    * (
                        2 * cy * ml
                        + sqrt(
                            -(bl**4)
                            + 2 * bl**2 * cx**2
                            + 2 * bl**2 * cy**2
                            + 2 * bl**2 * cz**2
                            + 2 * bl**2 * ml**2
                            - cx**4
                            - 2 * cx**2 * cy**2
                            - 2 * cx**2 * cz**2
                            + 2 * cx**2 * ml**2
                            - cy**4
                            - 2 * cy**2 * cz**2
                            + 2 * cy**2 * ml**2
                            - cz**4
                            - 2 * cz**2 * ml**2
                            - ml**4
                        )
                    )
                    * sqrt(
                        -(bl**4)
                        + 2 * bl**2 * cx**2
                        + 2 * bl**2 * cy**2
                        + 2 * bl**2 * cz**2
                        + 2 * bl**2 * ml**2
                        - cx**4
                        - 2 * cx**2 * cy**2
                        - 2 * cx**2 * cz**2
                        + 2 * cx**2 * ml**2
                        - cy**4
                        - 2 * cy**2 * cz**2
                        + 2 * cy**2 * ml**2
                        - cz**4
                        - 2 * cz**2 * ml**2
                        - ml**4
                    )
                    + (-(bl**2) + cx**2 + 2 * cx * ml + cy**2 + cz**2 + ml**2)
                    * (
                        bl**2 * cy
                        - cx**2 * cy
                        - cy**3
                        - cy * cz**2
                        + cy * ml**2
                        + ml
                        * sqrt(
                            -(bl**4)
                            + 2 * bl**2 * cx**2
                            + 2 * bl**2 * cy**2
                            + 2 * bl**2 * cz**2
                            + 2 * bl**2 * ml**2
                            - cx**4
                            - 2 * cx**2 * cy**2
                            - 2 * cx**2 * cz**2
                            + 2 * cx**2 * ml**2
                            - cy**4
                            - 2 * cy**2 * cz**2
                            + 2 * cy**2 * ml**2
                            - cz**4
                            - 2 * cz**2 * ml**2
                            - ml**4
                        )
                    )
                )
                / (
                    (
                        (
                            2 * cy * ml
                            + sqrt(
                                -(bl**4)
                                + 2 * bl**2 * cx**2
                                + 2 * bl**2 * cy**2
                                + 2 * bl**2 * cz**2
                                + 2 * bl**2 * ml**2
                                - cx**4
                                - 2 * cx**2 * cy**2
                                - 2 * cx**2 * cz**2
                                + 2 * cx**2 * ml**2
                                - cy**4
                                - 2 * cy**2 * cz**2
                                + 2 * cy**2 * ml**2
                                - cz**4
                                - 2 * cz**2 * ml**2
                                - ml**4
                            )
                        )
                        ** 2
                        + (-(bl**2) + cx**2 + 2 * cx * ml + cy**2 + cz**2 + ml**2) ** 2
                    )
                    * sqrt(
                        -(bl**4)
                        + 2 * bl**2 * cx**2
                        + 2 * bl**2 * cy**2
                        + 2 * bl**2 * cz**2
                        + 2 * bl**2 * ml**2
                        - cx**4
                        - 2 * cx**2 * cy**2
                        - 2 * cx**2 * cz**2
                        + 2 * cx**2 * ml**2
                        - cy**4
                        - 2 * cy**2 * cz**2
                        + 2 * cy**2 * ml**2
                        - cz**4
                        - 2 * cz**2 * ml**2
                        - ml**4
                    )
                ),
                4
                * cz
                * (
                    -(
                        2 * cy * ml
                        + sqrt(
                            -(bl**4)
                            + 2 * bl**2 * cx**2
                            + 2 * bl**2 * cy**2
                            + 2 * bl**2 * cz**2
                            + 2 * bl**2 * ml**2
                            - cx**4
                            - 2 * cx**2 * cy**2
                            - 2 * cx**2 * cz**2
                            + 2 * cx**2 * ml**2
                            - cy**4
                            - 2 * cy**2 * cz**2
                            + 2 * cy**2 * ml**2
                            - cz**4
                            - 2 * cz**2 * ml**2
                            - ml**4
                        )
                    )
                    * sqrt(
                        -(bl**4)
                        + 2 * bl**2 * cx**2
                        + 2 * bl**2 * cy**2
                        + 2 * bl**2 * cz**2
                        + 2 * bl**2 * ml**2
                        - cx**4
                        - 2 * cx**2 * cy**2
                        - 2 * cx**2 * cz**2
                        + 2 * cx**2 * ml**2
                        - cy**4
                        - 2 * cy**2 * cz**2
                        + 2 * cy**2 * ml**2
                        - cz**4
                        - 2 * cz**2 * ml**2
                        - ml**4
                    )
                    + (bl**2 - cx**2 - cy**2 - cz**2 - ml**2)
                    * (-(bl**2) + cx**2 + 2 * cx * ml + cy**2 + cz**2 + ml**2)
                )
                / (
                    (
                        (
                            2 * cy * ml
                            + sqrt(
                                -(bl**4)
                                + 2 * bl**2 * cx**2
                                + 2 * bl**2 * cy**2
                                + 2 * bl**2 * cz**2
                                + 2 * bl**2 * ml**2
                                - cx**4
                                - 2 * cx**2 * cy**2
                                - 2 * cx**2 * cz**2
                                + 2 * cx**2 * ml**2
                                - cy**4
                                - 2 * cy**2 * cz**2
                                + 2 * cy**2 * ml**2
                                - cz**4
                                - 2 * cz**2 * ml**2
                                - ml**4
                            )
                        )
                        ** 2
                        + (-(bl**2) + cx**2 + 2 * cx * ml + cy**2 + cz**2 + ml**2) ** 2
                    )
                    * sqrt(
                        -(bl**4)
                        + 2 * bl**2 * cx**2
                        + 2 * bl**2 * cy**2
                        + 2 * bl**2 * cz**2
                        + 2 * bl**2 * ml**2
                        - cx**4
                        - 2 * cx**2 * cy**2
                        - 2 * cx**2 * cz**2
                        + 2 * cx**2 * ml**2
                        - cy**4
                        - 2 * cy**2 * cz**2
                        + 2 * cy**2 * ml**2
                        - cz**4
                        - 2 * cz**2 * ml**2
                        - ml**4
                    )
                ),
            ],
        ]
    )


def jacobian_platform_leg_attach(branch_position, roll, pitch, yaw, T_motor_world=None):
    ax, ay, az = branch_position
    # The Jacobian of the platform leg attachment point in the head frame
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    jac_world = np.array(
        [
            [
                1,
                0,
                0,
                ay * (sp * cr * cy + sr * sy) - az * (sp * sr * cy - sy * cr),
                (-ax * sp + ay * sr * cp + az * cp * cr) * cy,
                -ax * sy * cp
                - ay * (sp * sr * sy + cr * cy)
                - az * (sp * sy * cr - sr * cy),
            ],
            [
                0,
                1,
                0,
                ay * (sp * sy * cr - sr * cy) - az * (sp * sr * sy + cr * cy),
                (-ax * sp + ay * sr * cp + az * cp * cr) * sy,
                ax * cp * cy
                + ay * (sp * sr * cy - sy * cr)
                + az * (sp * cr * cy + sr * sy),
            ],
            [
                0,
                0,
                1,
                (ay * cr - az * sr) * cp,
                -ax * cp - ay * sp * sr - az * sp * cr,
                0,
            ],
        ]
    )
    if T_motor_world is None:
        return jac_world
    else:
        jac_motor = T_motor_world @ np.vstack([jac_world, np.zeros(6)])
        return jac_motor[:3, :]


def ik_platform_leg_attach(
    attach_head, platform_world, roll, pitch, yaw, T_motor_world=None
):
    ax, ay, az = attach_head
    px, py, pz = platform_world

    # The inverse kinematics of the platform branch attachment point in the head frame
    cp = np.cos(pitch)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    leg_attach_world = np.array(
        [
            [
                ax * cp * cy
                + ay * (sp * sr * cy - sy * cr)
                + az * (sp * cr * cy + sr * sy)
                + px
            ],
            [
                ax * sy * cp
                + ay * (sp * sr * sy + cr * cy)
                + az * (sp * sy * cr - sr * cy)
                + py
            ],
            [-ax * sp + ay * sr * cp + az * cp * cr + pz],
        ]
    )

    if T_motor_world is None:
        return leg_attach_world
    else:
        leg_attach_motor = T_motor_world @ np.vstack([leg_attach_world, np.ones(1)])
        return leg_attach_motor[:3, 0]


def compute_joint_angles(px, py, pz, roll, pitch, yaw):
    T_world_target = (
        T_world_head
        @ tf.translation_matrix((px, py, pz))
        @ tf.euler_matrix(roll, pitch, yaw)
    )
    joints = {}
    for motor in motors:
        T_target_branch = tf.translation_matrix(motor["branch_position"])
        T_motor_branch = (
            np.linalg.inv(motor["T_world_motor"]) @ T_world_target @ T_target_branch
        )
        branch_position = T_motor_branch[:3, 3]
        try:
            solutions = ik_branch(*branch_position)
        except ValueError as e:
            print(f"Error in IK for motor {motor['name']}: {e}")
            break
        joints[motor["name"]] = placo.wrap_angle(
            motor["offset"] + solutions[motor["solution"]]
        )
    return joints, T_world_target


def compute_jacobian(T_world_head_current):
    T_headz_head = np.linalg.inv(T_world_head) @ T_world_head_current
    roll, pitch, yaw = tf.euler_from_matrix(T_headz_head[:3, :3], axes="sxyz")
    jacobian = np.zeros((len(motors), 6))
    for i, motor in enumerate(motors):
        T_target_branch = tf.translation_matrix(motor["branch_position"])
        T_motor_world = np.linalg.inv(motor["T_world_motor"])
        T_motor_branch = T_motor_world @ T_world_head_current @ T_target_branch
        branch_position = T_motor_branch[:3, 3]
        try:
            jac = jacobian_branch(*branch_position)[
                motor["solution"]
            ] @ jacobian_platform_leg_attach(
                motor["branch_position"], roll, pitch, yaw, T_motor_world
            )
            # print(motor["branch_position"])
            jacobian[i, :] = jac
        except ValueError as e:
            print(f"Error in Jacobian for motor {motor['name']}: {e}")
            break
    print("Jacobian computed:", jacobian)
    return np.array(jacobian)


def compute_joint_angles_partial_jac(T_world_head_current, error_p, error_rpy):
    T_headz_head = np.linalg.inv(T_world_head) @ T_world_head_current
    roll, pitch, yaw = tf.euler_from_matrix(T_headz_head[:3, :3], axes="sxyz")
    px, py, pz = T_world_head_current[:3, 3]
    joints = {}
    for i, motor in enumerate(motors):
        T_target_branch = tf.translation_matrix(motor["branch_position"])
        T_motor_branch = (
            np.linalg.inv(motor["T_world_motor"])
            @ T_world_head_current
            @ T_target_branch
        )
        branch_position = T_motor_branch[:3, 3]
        try:
            jac_leg_attach_motor = jacobian_platform_leg_attach(
                motor["branch_position"],
                roll,
                pitch,
                yaw,
                T_motor_world=np.linalg.inv(motor["T_world_motor"]),
            )
            new_branch_position = (
                branch_position
                + jac_leg_attach_motor @ np.concatenate((error_p, error_rpy))
            )
            joints[motor["name"]] = (
                ik_branch(*new_branch_position)[motor["solution"]] + motor["offset"]
            )
        except ValueError as e:
            print(f"Error in Jacobian for motor {motor['name']}: {e}")
            break
    return joints


def compute_joint_angles_partial_jac1(T_world_head_current, T_world_target):
    # T_world_target = T_world_head @ tf.translation_matrix((px, py, pz)) @ tf.euler_matrix(roll, pitch, yaw)
    roll, pitch, yaw = tf.euler_from_matrix(T_world_target[:3, :3], axes="sxyz")
    roll_current, pitch_current, yaw_current = tf.euler_from_matrix(
        T_world_head_current[:3, :3], axes="sxyz"
    )
    joints = {}
    for motor in motors:
        T_motor_world = np.linalg.inv(motor["T_world_motor"])

        try:
            branch_position_target = ik_platform_leg_attach(
                motor["branch_position"],
                T_world_target[:3, 3],
                roll,
                pitch,
                yaw,
                T_motor_world,
            ).flatten()

            branch_position_current = ik_platform_leg_attach(
                motor["branch_position"],
                T_world_head_current[:3, 3],
                roll_current,
                pitch_current,
                yaw_current,
                T_motor_world,
            ).flatten()

            # d_motor_joint_t = ik_branch(*branch_position_target)[motor["solution"]] + motor["offset"]
            # d_motor_joint_c = ik_branch(*branch_position_current)[motor["solution"]] + motor["offset"]
            # joints[motor["name"]] = d_motor_joint_t - d_motor_joint_c
            delta_branch_position = branch_position_target - branch_position_current
            d_motor_joint = (
                jacobian_branch(*branch_position_current)[motor["solution"]]
                @ delta_branch_position
            )
            joints[motor["name"]] = d_motor_joint
        except ValueError as e:
            print(f"Error in IK for motor {motor['name']}: {e}")
            break
    return joints


def compute_full_analytic_ik(px, py, pz, roll, pitch, yaw):
    T_world_target = (
        T_world_head
        @ tf.translation_matrix((px, py, pz))
        @ tf.euler_matrix(roll, pitch, yaw)
    )
    joints = {}
    for motor in motors:
        T_motor_world = np.linalg.inv(motor["T_world_motor"])
        branch_position = ik_platform_leg_attach(
            motor["branch_position"],
            T_world_target[:3, 3],
            roll,
            pitch,
            yaw,
            T_motor_world,
        ).flatten()
        try:
            solutions = ik_branch(*branch_position)
        except ValueError as e:
            print(f"Error in IK for motor {motor['name']}: {e}")
            break
        joints[motor["name"]] = placo.wrap_angle(
            motor["offset"] + solutions[motor["solution"]]
        )
    return joints, T_world_target


joints, _ = compute_joint_angles(*([0.0] * 6))

t = 0
while True:
    viz.display(robot.state.q)

    # Up and down
    # T_world_target = T_world_head @ tf.translation_matrix((0, 0, np.sin(t*3)*0.01))

    # Rolling
    # --- Tkinter GUI for position and Euler angles ---

    if t == 0:
        root = tk.Tk()
        root.title("Target Position and Orientation")

        pos_vars = [tk.DoubleVar(value=0.0) for _ in range(3)]
        rpy_vars = [tk.DoubleVar(value=0.0) for _ in range(3)]

        labels = ["X (m)", "Y (m)", "Z (m)", "Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
        for i, label in enumerate(labels):
            tk.Label(root, text=label).grid(row=i, column=0)
            var = pos_vars[i] if i < 3 else rpy_vars[i - 3]
            tk.Scale(
                root,
                variable=var,
                from_=-0.05 if i < 3 else -1.57,
                to=0.05 if i < 3 else 1.57,
                resolution=0.001,
                orient="horizontal",
                length=200,
            ).grid(row=i, column=1)

        # def update_gui():
    root.update()
    # root.after(10, update_gui)

    t += 0.02

    # Read GUI values
    px, py, pz = [v.get() for v in pos_vars]
    roll, pitch, yaw = [v.get() for v in rpy_vars]

    T_world_target = (
        T_world_head
        @ tf.translation_matrix((px, py, pz))
        @ tf.euler_matrix(roll, pitch, yaw)
    )
    joints, _ = compute_full_analytic_ik(px, py, pz, roll, pitch, yaw)
    jacobian = compute_jacobian(robot.get_T_world_frame("head"))

    if len(joints) != len(motors):
        print("joints IK failed for some motors, skipping iteration")
        continue

    if jacobian.shape[0] == len(motors):
        print("p:", 1 / np.linalg.svd(jacobian[:, :3], compute_uv=False))
        print("o:", 1 / np.linalg.svd(jacobian[:, 3:], compute_uv=False))
    else:
        continue

    dM = pin.SE3(robot.get_T_world_frame("head")).actInv(pin.SE3(T_world_target))
    # #print(robot.get_T_world_frame("head"), T_world_target)
    # # error = -pin.log6(dM)
    # # dq = jacobian @ error
    # # print(error, dq)

    # # pos_head = robot.get_T_world_frame("head")[:3, 3]
    # # pos_target = T_world_target[:3, 3]
    # # error_p = pos_target - pos_head
    # # dq = jacobian[:,:3] @ error_p
    # # print(error_p,  dq)

    # rpy_head = tf.euler_from_matrix(robot.get_T_world_frame("head")[:3, :3], axes='sxyz')
    # rpy_target = tf.euler_from_matrix(T_world_target[:3, :3], axes='sxyz')
    # rpy_error = np.array(rpy_target) - np.array(rpy_head)
    # error_rpy = np.array(rpy_error)
    # dq = jacobian[:,3:] @ error_rpy
    # pos_head = robot.get_T_world_frame("head")[:3, 3]
    # pos_target = T_world_target[:3, 3]
    # error_p = pos_target - pos_head
    # dq = dq + jacobian[:,:3] @ error_p
    # print("pos error:", error_p)
    # print("rpy error:", error_rpy)
    # print("dq:", dq)

    # # dq = jacobian[:,:3] @ np.array([0.01,0,0])
    # # print("dq:", dq)

    error_p = dM.translation
    error_rpy = np.array(tf.euler_from_matrix(dM.rotation, axes="sxyz"))
    dq = jacobian @ np.concatenate((error_p, error_rpy))
    print(error_p, error_rpy, dq)

    joints_dq = dq * 0.1
    joints = {
        name: value + joints_dq[i] for i, (name, value) in enumerate(joints.items())
    }

    # for i, motor in enumerate(motors):
    #     if motor["name"] in joints:
    #         joints[motor["name"]] += joints_dq[i]
    #     else:
    #         print(f"Motor {motor['name']} not found in joints, skipping update")

    # rpy_head = tf.euler_from_matrix(robot.get_T_world_frame("head")[:3, :3], axes='sxyz')
    # rpy_target = tf.euler_from_matrix(T_world_target[:3, :3], axes='sxyz')
    # rpy_error = np.array(rpy_target) - np.array(rpy_head)
    # error_rpy = np.array(rpy_error)
    # pos_head = robot.get_T_world_frame("head")[:3, 3]
    # pos_target = T_world_target[:3, 3]
    # error_p = pos_target - pos_head
    # joints = compute_joint_angles_partial_jac(robot.get_T_world_frame("head"), error_p , error_rpy)
    # print(joints, "error_p:", error_p, "error_rpy:", error_rpy)

    # joints, _ = compute_full_analytic_ik(px, py, pz, roll, pitch, yaw)

    # d_joints = compute_joint_angles_partial_jac1(robot.get_T_world_frame("head"), T_world_target)
    # joints = {name: value + 0.2* d_joints[name] for name, value in joints.items() if name in d_joints}

    solver.fk([0.0] + list(joints.values()))

    for k in range(1, 7):
        robot_frame_viz(robot, f"{k}", scale=0.2)

    robot_frame_viz(robot, "closing_3_2", scale=0.2)
    T_world_branch1 = ik_platform_leg_attach(
        motors[2]["branch_position"],
        T_world_target[:3, 3],
        roll,
        pitch,
        yaw,
        np.eye(4),  # Assuming identity for T_motor_world
    )
    T_world_branch1 = tf.translation_matrix(T_world_branch1.flatten())
    frame_viz("branch_3", T_world_branch1, scale=0.2, opacity=0.5)

    robot_frame_viz(robot, "head", scale=0.5)
    frame_viz("target", T_world_target, scale=0.75, opacity=0.5)

    frame_viz("head_world", T_world_head, scale=0.5, opacity=0.5)
    robot.update_kinematics()
    time.sleep(0.02)
