"""Generate kinematics data from URDF using Placo as preprocessing.

The analytical kinematics need information from the URDF. This files computes the information and writes it in a .json file.
"""

import json
from importlib.resources import files
from typing import Any, Dict

import numpy as np  # noqa: D100
from placo_utils.tf import tf

import reachy_mini
from reachy_mini.kinematics.placo_kinematics import PlacoKinematics


def get_data() -> Dict[str, Any]:
    """Generate the urdf_kinematics.json file."""
    urdf_root_path: str = str(
        files(reachy_mini).joinpath("descriptions/reachy_mini/urdf")
    )

    placo_kinematics = PlacoKinematics(urdf_root_path, 0.02)
    robot = placo_kinematics.robot

    placo_kinematics.fk(np.array([0.0] * 7), no_iterations=20)
    robot.update_kinematics()

    # Measuring lengths for the arm and branch (constants could be used)
    T_world_head_home = robot.get_T_world_frame("head").copy()
    T_world_1 = robot.get_T_world_frame("stewart_1")
    T_world_arm1 = robot.get_T_world_frame("passive_1_link_x")
    T_1_arm1 = np.linalg.inv(T_world_1) @ T_world_arm1
    arm_z = T_1_arm1[2, 3]
    motor_arm_length = np.linalg.norm(T_1_arm1[:2, 3])

    T_world_branch1 = robot.get_T_world_frame("closing_1_2")
    T_arm1_branch1 = np.linalg.inv(T_world_arm1) @ T_world_branch1
    rod_length = np.linalg.norm(T_arm1_branch1[:3, 3])

    motors = [
        {
            "name": "stewart_1",
            "branch_frame": "closing_1_2",
            "offset": 0,
            "solution": 0,
        },
        {
            "name": "stewart_2",
            "branch_frame": "closing_2_2",
            "offset": 0,
            "solution": 1,
        },
        {
            "name": "stewart_3",
            "branch_frame": "closing_3_2",
            "offset": 0,
            "solution": 0,
        },
        {
            "name": "stewart_4",
            "branch_frame": "closing_4_2",
            "offset": 0,
            "solution": 1,
        },
        {
            "name": "stewart_5",
            "branch_frame": "closing_5_2",
            "offset": 0,
            "solution": 0,
        },
        {
            "name": "stewart_6",
            "branch_frame": "passive_7_link_y",
            "offset": 0,
            "solution": 1,
        },
    ]

    for motor in motors:
        T_world_branch = robot.get_T_world_frame(motor["branch_frame"])
        T_head_branch = np.linalg.inv(T_world_head_home) @ T_world_branch
        T_world_motor = robot.get_T_world_frame(motor["name"]) @ tf.translation_matrix(
            (0, 0, arm_z)
        )
        motor["T_motor_world"] = np.linalg.inv(T_world_motor).tolist()
        motor["branch_position"] = T_head_branch[:3, 3].tolist()
        motor["limits"] = robot.get_joint_limits(motor["name"]).tolist()

    data = {
        "motor_arm_length": motor_arm_length,
        "rod_length": rod_length,
        "head_z_offset": placo_kinematics.head_z_offset,
        "motors": motors,
    }

    return data


def main() -> None:
    """Generate the urdf_kinematics.json file."""
    assets_root_path: str = str(files(reachy_mini).joinpath("assets/"))
    data = get_data()
    print(assets_root_path + "/" + "kinematics_data.json")
    with open(assets_root_path + "/" + "kinematics_data.json", "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
