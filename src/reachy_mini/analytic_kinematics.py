import numpy as np
import pinocchio as pin
from .placo_kinematics import PlacoKinematics
import placo
from placo_utils.tf import tf


class ReachyMiniAnalyticKinematics:
    def __init__(self, urdf_path=None, robot=None):

        if urdf_path is None and robot is None:
            raise ValueError("Either urdf_path or robot must be provided.")
        if robot is not None:
            self.robot = robot
        elif urdf_path is not None:
            solver = PlacoKinematics(urdf_path, 0.02)
            self.robot = solver.robot

        self.robot.update_kinematics()

        self.motors = [
            {"name": "1", "branch_frame": "closing_1_2", "offset": 0, "solution": 0},
            {"name": "2", "branch_frame": "closing_2_2", "offset": 0, "solution": 1},
            {"name": "3", "branch_frame": "closing_3_2", "offset": 0, "solution": 0},
            {"name": "4", "branch_frame": "closing_4_2", "offset": 0, "solution": 1},
            {"name": "5", "branch_frame": "closing_5_2", "offset": 0, "solution": 0},
            {
                "name": "6",
                "branch_frame": "passive_7_link_y",
                "offset": 0,
                "solution": 1,
            },
        ]

        # Measuring lengths for the arm and branch (constants could be used)
        self.T_world_head_home = self.robot.get_T_world_frame("head").copy()
        T_world_1 = self.robot.get_T_world_frame("1")
        T_world_arm1 = self.robot.get_T_world_frame("passive_1_link_x")

        T_1_arm1 = np.linalg.inv(T_world_1) @ T_world_arm1
        arm_z = T_1_arm1[2, 3]
        self.servo_arm_length = np.linalg.norm(T_1_arm1[:2, 3])
        T_world_branch1 = self.robot.get_T_world_frame("closing_1_2")
        T_arm1_branch1 = np.linalg.inv(T_world_arm1) @ T_world_branch1
        self.branch_length = np.linalg.norm(T_arm1_branch1[:3, 3])

        # Finding the 6 branches position in the head frame, and building T_world_motor frame
        # T_world_motor is a frame where the motor is located, correcting for the arm_z offset
        for motor in self.motors:
            T_world_branch = self.robot.get_T_world_frame(motor["branch_frame"])
            T_head_branch = np.linalg.inv(self.T_world_head_home) @ T_world_branch
            T_world_motor = self.robot.get_T_world_frame(
                motor["name"]
            ) @ tf.translation_matrix((0, 0, arm_z))
            motor["T_motor_world"] = np.linalg.inv(T_world_motor)
            motor["branch_position"] = T_head_branch[:3, 3]

    def ik_motor_to_branch(self, branch_attachment_platform, solution=0):
        """
        Inverse kinematics for the branch attachment platform to the motor angles.
        This function computes the inverse kinematics for the branch attachment platform
        to the motor angles based on the branch attachment platform position and the
        servo arm and branch lengths.

        Args:
            branch_attachment_platform (tuple): A tuple of three floats representing the
                x, y, z coordinates of the branch attachment platform in the head frame.
            solution (int): The solution index for the inverse kinematics. It can be either
                0 or 1, representing the two possible solutions for the inverse kinematics.
        Returns:
            float: The angle in radians for the motor that corresponds to the branch
            attachment platform position.
        """

        px, py, pz = branch_attachment_platform
        rs = self.servo_arm_length
        rp = self.branch_length

        # two possile solutions for the inverse kinematics
        if solution == 0:
            return 2 * np.atan2(
                (
                    2 * py * rs
                    - np.sqrt(
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
                ),
                (px**2 + 2 * px * rs + py**2 + pz**2 - rp**2 + rs**2),
            )
        else:
            return 2 * np.atan2(
                (
                    2 * py * rs
                    + np.sqrt(
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
                ),
                (px**2 + 2 * px * rs + py**2 + pz**2 - rp**2 + rs**2),
            )

    def jacobian_motor_to_branch(self, branch_attachment_platform, solution=0):
        """
        Calculates the Jacobian matrix for the platform branch attachment point in the motor frame.
        
        Args:
            branch_attachment_platform (tuple): A tuple of three floats representing the
                x, y, z coordinates of the branch attachment platform in the head frame.
            solution (int): The solution index for the inverse kinematics. It can be either
                0 or 1, representing the two possible solutions for the inverse kinematics.
        Returns:
            np.ndarray: A 3x3 Jacobian matrix representing the relationship between the
            branch attachment platform position and the motor angles.

        """
        cx, cy, cz = branch_attachment_platform
        ml = self.servo_arm_length
        bl = self.branch_length

        if solution == 0:
            return np.array(
                [
                    4
                    * (
                        cx
                        * (-(bl**2) + cx**2 + cy**2 + cz**2 - ml**2)
                        * (-(bl**2) + cx**2 + 2 * cx * ml + cy**2 + cz**2 + ml**2)
                        - (cx + ml)
                        * (
                            2 * cy * ml
                            - np.sqrt(
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
                        * np.sqrt(
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
                                - np.sqrt(
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
                            + (-(bl**2) + cx**2 + 2 * cx * ml + cy**2 + cz**2 + ml**2)
                            ** 2
                        )
                        * np.sqrt(
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
                            - np.sqrt(
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
                        * np.sqrt(
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
                            * np.sqrt(
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
                                - np.sqrt(
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
                            + (-(bl**2) + cx**2 + 2 * cx * ml + cy**2 + cz**2 + ml**2)
                            ** 2
                        )
                        * np.sqrt(
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
                            - np.sqrt(
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
                        * np.sqrt(
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
                                - np.sqrt(
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
                            + (-(bl**2) + cx**2 + 2 * cx * ml + cy**2 + cz**2 + ml**2)
                            ** 2
                        )
                        * np.sqrt(
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
                ]
            )
        else:
            return np.array(
                [
                    4
                    * (
                        cx
                        * (bl**2 - cx**2 - cy**2 - cz**2 + ml**2)
                        * (-(bl**2) + cx**2 + 2 * cx * ml + cy**2 + cz**2 + ml**2)
                        - (cx + ml)
                        * (
                            2 * cy * ml
                            + np.sqrt(
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
                        * np.sqrt(
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
                                + np.sqrt(
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
                            + (-(bl**2) + cx**2 + 2 * cx * ml + cy**2 + cz**2 + ml**2)
                            ** 2
                        )
                        * np.sqrt(
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
                            + np.sqrt(
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
                        * np.sqrt(
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
                            * np.sqrt(
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
                                + np.sqrt(
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
                            + (-(bl**2) + cx**2 + 2 * cx * ml + cy**2 + cz**2 + ml**2)
                            ** 2
                        )
                        * np.sqrt(
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
                            + np.sqrt(
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
                        * np.sqrt(
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
                                + np.sqrt(
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
                            + (-(bl**2) + cx**2 + 2 * cx * ml + cy**2 + cz**2 + ml**2)
                            ** 2
                        )
                        * np.sqrt(
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
                ]
            )

    def jacobian_platform_to_branch(
        self, branch_position, roll, pitch, yaw, T_motor_world=None
    ):
        """
        Calculates the Jacobian matrix for the platform branch attachment point in the motor (if provided) frame.

        Args:
            branch_position (np.ndarray): The position of the branch attachment point in the head frame,
                represented as a 3-element numpy array.
            roll (float): The roll angle in radians.
            pitch (float): The pitch angle in radians.
            yaw (float): The yaw angle in radians.
            T_motor_world (np.ndarray, optional): The transformation matrix from the motor frame to the
                world frame, represented as a 4x4 numpy array. If None, the result will be in the world frame.

        Returns:
            np.ndarray: The Jacobian matrix for the platform branch attachment point in the motor frame (
            if T_motor_world is provided) or in the world frame (if T_motor_world is None), represented as a
                3x6 numpy array.
        """
        ax, ay, az = branch_position
        # The Jacobian of the platform branch attachment point in the head frame
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

    def ik_platform_to_branch(
        self, attach_head, platform_world, roll, pitch, yaw, T_motor_world=None
    ):
        """
            Calculates the inverse kinematics for the platform branch attachment point in the motor frame (if T_motor_world is provided else in world frame).

        Args:
            attach_head (np.ndarray): The attachment point in the head frame, represented as a
                3-element numpy array.
            platform_world (np.ndarray): The platform position in the world frame, represented as a
                3-element numpy array.
            roll (float): The roll angle in radians.
            pitch (float): The pitch angle in radians.
            yaw (float): The yaw angle in radians.
            T_motor_world (np.ndarray, optional): The transformation matrix from the motor frame to the
                world frame, represented as a 4x4 numpy array. If None, the result
                will be in the world frame.

        Returns:
            np.ndarray: The position of the leg attachment point in the motor frame (if T_motor
            world is provided) or in the world frame (if T_motor_world is None), represented as a
                3-element numpy array.
        """

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

    def ik(self, T_world_target):
        """
        Calculates the inverse kinematics for the Reachy Mini robot to reach a target position and orientation.

        Args:
            T_world_target (np.ndarray): The target transformation matrix in the world frame, represented as
                a 4x4 numpy array.

        Returns:
            dict: A dictionary containing the joint angles for each motor, where keys are motor names and
                values are the corresponding joint angles in radians.
        """
        roll, pitch, yaw = tf.euler_from_matrix(T_world_target[:3, :3], axes="sxyz")
        joints = {}
        for motor in self.motors:
            T_motor_world = motor["T_motor_world"]
            try:
                branch_position = self.ik_platform_to_branch(
                    motor["branch_position"],
                    T_world_target[:3, 3],
                    roll,
                    pitch,
                    yaw,
                    T_motor_world,
                ).flatten()
            except Exception as e:
                print(
                    f"Error in IK for branch attach IK for the branch {motor['name']}: {e}"
                )
                return {}
            try:
                solution = (
                    self.ik_motor_to_branch(branch_position, motor["solution"])
                    + motor["offset"]
                )
            except Exception as e:
                print(f"Error in IK for motor {motor['name']}: {e}")
                return {}
            joints[motor["name"]] = placo.wrap_angle(solution)
        return joints

    def jacobian(self, T_world_head_current):
        """
        Calculates the head Jacobian matrix for the Reachy Mini robot.
        The convention is local world aligned.

        The first three rows correspond to the linear velocity of the head,
        and the last three rows correspond to the angular velocity in the roll, pitch, and yaw axes.

        Args:
            T_world_head_current (np.ndarray): The current transformation matrix of the head in the world
                                               frame, represented as a 4x4 numpy array.
        Returns:
            np.ndarray: The Jacobian matrix of shape (n_motors, 6), where n_motors is the number of motors in the head.
                        The first three rows correspond to the linear velocity of the head,
                        and the last three rows correspond to the angular velocity in the roll, pitch, and yaw axes.
        """
        T_headz_head = np.linalg.inv(self.T_world_head_home) @ T_world_head_current
        roll, pitch, yaw = tf.euler_from_matrix(T_headz_head[:3, :3], axes="sxyz")
        jacobian = np.zeros((len(self.motors), 6))
        for i, motor in enumerate(self.motors):
            T_target_branch = tf.translation_matrix(motor["branch_position"])
            T_motor_world = motor["T_motor_world"]
            T_motor_branch = T_motor_world @ T_world_head_current @ T_target_branch
            branch_position = T_motor_branch[:3, 3]
            try:
                jac = self.jacobian_motor_to_branch(
                    branch_position, motor["solution"]
                ) @ self.jacobian_platform_to_branch(
                    motor["branch_position"], roll, pitch, yaw, T_motor_world
                )
                jacobian[i, :] = jac
            except Exception as e:
                print(f"Error in Jacobian for motor {motor['name']}: {e}")
                break
        return np.array(jacobian)
