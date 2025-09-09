import numpy as np
import reachy_mini_kinematics as rk
from placo_utils.tf import tf

from reachy_mini.kinematics import PlacoKinematics


class CPPAnalyticKinematics:
    """Reachy Mini Analytic Kinematics class, implemented in C++ with python bindings."""

    def __init__(self, urdf_path: str):
        """Initialize."""
        self.urdf_path = urdf_path

        # TODO get rid of the PlacoKinematics dependency
        # Store the values, or use a different urdf parser ?
        self.placo_kinematics = PlacoKinematics(urdf_path, 0.02)
        self.robot = self.placo_kinematics.robot

        self.placo_kinematics.fk([0.0] * 7, no_iterations=20)
        self.robot.update_kinematics()

        # Measuring lengths for the arm and branch (constants could be used)
        self.T_world_head_home = self.robot.get_T_world_frame("head").copy()
        T_world_1 = self.robot.get_T_world_frame("1")
        T_world_arm1 = self.robot.get_T_world_frame("passive_1_link_x")
        T_1_arm1 = np.linalg.inv(T_world_1) @ T_world_arm1
        arm_z = T_1_arm1[2, 3]
        self.motor_arm_length = np.linalg.norm(T_1_arm1[:2, 3])

        T_world_branch1 = self.robot.get_T_world_frame("closing_1_2")
        T_arm1_branch1 = np.linalg.inv(T_world_arm1) @ T_world_branch1
        self.rod_length = np.linalg.norm(T_arm1_branch1[:3, 3])

        self.kin = rk.Kinematics(self.motor_arm_length, self.rod_length)

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

        for motor in self.motors:
            T_world_branch = self.robot.get_T_world_frame(motor["branch_frame"])
            T_head_branch = np.linalg.inv(self.T_world_head_home) @ T_world_branch
            T_world_motor = self.robot.get_T_world_frame(
                motor["name"]
            ) @ tf.translation_matrix((0, 0, arm_z))
            motor["T_motor_world"] = np.linalg.inv(T_world_motor)
            motor["branch_position"] = T_head_branch[:3, 3]
            motor["limits"] = self.robot.get_joint_limits(motor["name"])

        for motor in self.motors:
            self.kin.add_branch(
                np.array(motor["branch_position"]),
                np.linalg.inv(motor["T_motor_world"]),
                1 if motor["solution"] else -1,
            )

        initial_T_world_platform = np.eye(4)
        initial_T_world_platform[:3, 3][2] += self.placo_kinematics.head_z_offset
        self.kin.reset_forward_kinematics(initial_T_world_platform)

    def ik(
        self,
        pose: np.ndarray,
        body_yaw: float = 0.0,
        check_collision: bool = False,
        no_iterations: int = 0,
    ):
        """check_collision and no_iterations are not used by CPPAnalyticKinematics.

        We keep them for compatibility with the other kinematics engines
        """
        # return self.placo_kinematics.ik(pose)
        _pose = pose.copy()
        _pose[:3, 3][2] += self.placo_kinematics.head_z_offset
        return [body_yaw] + list(self.kin.inverse_kinematics(_pose))

    def fk(
        self,
        joint_angles: list,
        check_collision: bool = False,
        no_iterations: int = 3,
    ):
        """check_collision and no_iterations are not used by CPPAnalyticKinematics.

        Not implemented in C++ version, using PlacoKinematics's FK instead.
        """
        _joint_angles = joint_angles[1:]
        for _ in range(no_iterations):
            T_world_platform = self.kin.forward_kinematics(_joint_angles)
        T_world_platform[:3, 3][2] -= self.placo_kinematics.head_z_offset
        return T_world_platform
        return self.placo_kinematics.fk(joint_angles)


if __name__ == "__main__":
    cpp_kin = CPPAnalyticKinematics(
        urdf_path="../descriptions/reachy_mini/urdf/robot.urdf"
    )
    print("Motor arm length:", cpp_kin.motor_arm_length)
    print("Rod length:", cpp_kin.rod_length)

    pose = np.eye(4)

    # SLEEP_HEAD_POSE = np.array(
    #     [
    #         [0.911, 0.004, 0.413, -0.021],
    #         [-0.004, 1.0, -0.001, 0.001],
    #         [-0.413, -0.001, 0.911, -0.044],
    #         [0.0, 0.0, 0.0, 1.0],
    #     ]
    # )

    print(np.around(cpp_kin.ik(pose), 3))
