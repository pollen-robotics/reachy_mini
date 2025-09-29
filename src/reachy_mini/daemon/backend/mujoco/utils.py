"""Mujoco utilities for Reachy Mini.

This module provides utility functions for working with MuJoCo models, including
homogeneous transformation matrices, joint positions, and actuator names.
"""

from typing import Annotated

import mujoco
import numpy as np
import numpy.typing as npt
from mujoco._structs import MjData, MjModel
from scipy.spatial.transform import Rotation as R


def get_homogeneous_matrix_from_euler(
    position: tuple[float, float, float] = (0, 0, 0),  # (x, y, z) meters
    euler_angles: tuple[float, float, float] = (0, 0, 0),  # (roll, pitch, yaw)
    degrees: bool = False,
) -> Annotated[npt.NDArray[np.float64], (4, 4)]:
    """Return a homogeneous transformation matrix from position and Euler angles."""
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = R.from_euler(
        "xyz", euler_angles, degrees=degrees
    ).as_matrix()
    homogeneous_matrix[:3, 3] = position
    return homogeneous_matrix


def get_joint_qpos(model: MjModel, data: MjData, joint_name: str) -> float:
    """Return the qpos (rad) of a specified joint in the model."""
    # Get the joint id
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id == -1:
        raise ValueError(f"Joint '{joint_name}' not found.")

    # Get the address of the joint's qpos in the qpos array
    qpos_addr = model.jnt_qposadr[joint_id]

    # Get the qpos value
    qpos: float = data.qpos[qpos_addr]
    return qpos


def get_joint_id_from_name(model: MjModel, name: str) -> int:
    """Return the id of a specified joint."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)  # type: ignore


def get_joint_addr_from_name(model: MjModel, name: str) -> int:
    """Return the address of a specified joint."""
    addr: int = model.joint(name).qposadr
    return addr


def get_actuator_names(model: MjModel) -> list[str]:
    """Return the list of the actuators names from the MuJoCo model."""
    actuator_names = [model.actuator(k).name for k in range(0, model.nu)]
    return actuator_names
