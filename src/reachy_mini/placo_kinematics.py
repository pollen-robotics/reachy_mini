"""Placo Kinematics for Reachy Mini.

This module provides the PlacoKinematics class for performing inverse and forward kinematics based on the Reachy Mini robot URDF using the Placo library.
"""

from typing import List

import numpy as np
import placo


class PlacoKinematics:
    """Placo Kinematics class for Reachy Mini.

    This class provides methods for inverse and forward kinematics using the Placo library and a URDF model of the Reachy Mini robot.
    """

    def __init__(self, urdf_path: str, dt: float = 0.02) -> None:
        """Initialize the PlacoKinematics class.

        Args:
            urdf_path (str): Path to the URDF file of the Reachy Mini robot.
            dt (float): Time step for the kinematics solver. Default is 0.02 seconds.

        """
        self.robot = placo.RobotWrapper(urdf_path, placo.Flags.ignore_collisions)

        self.ik_solver = placo.KinematicsSolver(self.robot)
        self.ik_solver.mask_fbase(True)

        self.fk_solver = placo.KinematicsSolver(self.robot)
        self.fk_solver.mask_fbase(True)

        # IK closing tasks
        ik_closing_tasks = []
        for i in range(1, 6):
            ik_closing_task = self.ik_solver.add_relative_position_task(
                f"closing_{i}_1", f"closing_{i}_2", np.zeros(3)
            )
            ik_closing_task.configure(f"closing_{i}", "hard", 1.0)
            ik_closing_tasks.append(ik_closing_task)

        # FK closing tasks
        fk_closing_tasks = []
        for i in range(1, 6):
            fk_closing_task = self.fk_solver.add_relative_position_task(
                f"closing_{i}_1", f"closing_{i}_2", np.zeros(3)
            )
            fk_closing_task.configure(f"closing_{i}", "hard", 1.0)
            fk_closing_tasks.append(fk_closing_task)

        self.joints_names = [
            "all_yaw",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
        ]

        self.head_z_offset = 0.177  # offset for the head height

        # IK head task
        self.head_starting_pose = np.eye(4)
        self.head_starting_pose[:3, 3][2] = self.head_z_offset
        self.head_frame = self.ik_solver.add_frame_task("head", self.head_starting_pose)
        self.head_frame.configure("head", "soft", 1.0, 1.0)

        self.head_frame.T_world_frame = self.head_starting_pose

        # regularization
        self.ik_yaw_joint_task = self.ik_solver.add_joints_task()
        self.ik_yaw_joint_task.set_joints({"all_yaw": 0})
        self.ik_yaw_joint_task.configure("joints", "soft", 5e-5)

        self.ik_joint_task = self.ik_solver.add_joints_task()
        self.ik_joint_task.set_joints({f"{k}": 0 for k in range(1, 7)})
        self.ik_joint_task.configure("joints", "soft", 1e-5)

        self.ik_solver.enable_velocity_limits(True)
        self.ik_solver.dt = dt

        # FK joint task
        self.head_joints_task = self.fk_solver.add_joints_task()
        self.head_joints_task.configure("joints", "soft", 1.0)

        # regularization
        self.fk_yaw_joint_task = self.fk_solver.add_joints_task()
        self.fk_yaw_joint_task.set_joints({"all_yaw": 0})
        self.fk_yaw_joint_task.configure("joints", "soft", 5e-5)

        self.fk_joint_task = self.fk_solver.add_joints_task()
        self.fk_joint_task.set_joints({f"{k}": 0 for k in range(1, 7)})
        self.fk_joint_task.configure("joints", "soft", 1e-5)

        # self.fk_solver.enable_velocity_limits(True)
        self.fk_solver.dt = dt

    def ik(self, pose: np.ndarray) -> List[float]:
        """Compute the inverse kinematics for the head for a given pose.

        Args:
            pose (np.ndarray): A 4x4 homogeneous transformation matrix

        Returns:
            List[float]: A list of joint angles for the head.

        """
        _pose = pose.copy()
        _pose[:3, 3][2] += self.head_z_offset  # offset the height of the head
        self.head_frame.T_world_frame = _pose
        for _ in range(10):
            try:
                self.ik_solver.solve(True)
                self.robot.update_kinematics()

            except RuntimeError as e:
                print(e)
                self.robot.reset()
                self.robot.update_kinematics()
                break

        joints = []
        for joint_name in self.joints_names:
            joint = self.robot.get_joint(joint_name)
            joints.append(joint)

        return joints

    def fk(self, joints_angles: List[float]) -> np.ndarray:
        """Compute the forward kinematics for the head given joint angles.

        Args:
            joints_angles (List[float]): A list of joint angles for the head.

        Returns:
            np.ndarray: A 4x4 homogeneous transformation matrix

        """
        self.head_joints_task.set_joints(
            {
                "all_yaw": joints_angles[0],
                "1": joints_angles[1],
                "2": joints_angles[2],
                "3": joints_angles[3],
                "4": joints_angles[4],
                "5": joints_angles[5],
                "6": joints_angles[6],
            }
        )

        for _ in range(10):
            try:
                self.fk_solver.solve(True)
                self.robot.update_kinematics()

            except RuntimeError as e:
                print(e)
                self.robot.reset()
                self.robot.update_kinematics()
                break

        T_world_head = self.robot.get_T_world_frame("head")
        T_world_head[:3, 3][2] -= self.head_z_offset  # offset the height of the head
        return T_world_head
