"""Mujoco utilities for Reachy Mini.

This module provides utility functions for working with MuJoCo models, including
homogeneous transformation matrices, joint positions, and actuator names.
"""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


def get_homogeneous_matrix_from_euler(
    position: tuple = (0, 0, 0),  # (x, y, z) meters
    euler_angles: tuple = (0, 0, 0),  # (roll, pitch, yaw)
    degrees: bool = False,
):
    """Return a homogeneous transformation matrix from position and Euler angles."""
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = R.from_euler(
        "xyz", euler_angles, degrees=degrees
    ).as_matrix()
    homogeneous_matrix[:3, 3] = position
    return homogeneous_matrix


def get_joint_qpos(model, data, joint_name) -> float:
    """Return the qpos (rad) of a specified joint in the model."""
    # Get the joint id
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)  # type: ignore
    if joint_id == -1:
        raise ValueError(f"Joint '{joint_name}' not found.")

    # Get the address of the joint's qpos in the qpos array
    qpos_addr = model.jnt_qposadr[joint_id]

    # Get the qpos value
    return data.qpos[qpos_addr]


def get_joint_id_from_name(model, name: str) -> int:
    """Return the id of a specified joint."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)  # type: ignore


def get_joint_addr_from_name(model, name: str) -> int:
    """Return the address of a specified joint."""
    return model.joint(name).qposadr


def get_actuator_names(model):
    """Return the list of the actuators names from the MuJoCo model."""
    actuator_names = [model.actuator(k).name for k in range(0, model.nu)]
    return actuator_names
