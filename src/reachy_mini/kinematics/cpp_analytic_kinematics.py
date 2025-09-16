"""Reachy Mini Analytic Kinematics class, implemented in C++ with python bindings."""

import json
from importlib.resources import files

import numpy as np  # noqa: D100
import reachy_mini_kinematics as rk

import reachy_mini

# Duplicated for now.
SLEEP_HEAD_POSE = np.array(
    [
        [0.911, 0.004, 0.413, -0.021],
        [-0.004, 1.0, -0.001, 0.001],
        [-0.413, -0.001, 0.911, -0.044],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


class CPPAnalyticKinematics:
    """Reachy Mini Analytic Kinematics class, implemented in C++ with python bindings."""

    def __init__(self):
        """Initialize."""
        assets_root_path: str = str(files(reachy_mini).joinpath("assets/"))
        data_path = assets_root_path + "/kinematics_data.json"
        data = json.load(open(data_path, "rb"))

        self.head_z_offset = data["head_z_offset"]

        self.kin = rk.Kinematics(data["motor_arm_length"], data["rod_length"])

        self.motors = data["motors"]
        for motor in self.motors:
            self.kin.add_branch(
                np.array(motor["branch_position"]),
                np.linalg.inv(motor["T_motor_world"]),
                1 if motor["solution"] else -1,
            )

        sleep_head_pose = SLEEP_HEAD_POSE
        sleep_head_pose[:3, 3][2] += self.head_z_offset
        self.kin.reset_forward_kinematics(sleep_head_pose)

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
        _pose = pose.copy()
        _pose[:3, 3][2] += self.head_z_offset
        return [body_yaw] + list(self.kin.inverse_kinematics(_pose))

    def fk(
        self,
        joint_angles: list,
        check_collision: bool = False,
        no_iterations: int = 3,
    ):
        """check_collision is not used by CPPAnalyticKinematics.

        For now, ignores the body yaw (first joint angle).
        """
        _joint_angles = joint_angles[1:]

        for _ in range(no_iterations):
            T_world_platform = self.kin.forward_kinematics(np.double(_joint_angles))

        T_world_platform[:3, 3][2] -= self.head_z_offset
        return T_world_platform
