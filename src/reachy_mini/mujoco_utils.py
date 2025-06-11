import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_homogeneous_matrix_from_euler(
    position: tuple = (0, 0, 0),  # (x, y, z)
    euler_angles: tuple = (0, 0, 0),  # (roll, pitch, yaw)
    degrees: bool = False,
):
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = R.from_euler("xyz", euler_angles, degrees=degrees).as_matrix()
    homogeneous_matrix[:3, 3] = position
    return homogeneous_matrix

def get_joint_qpos(model, data, joint_name):
    # Get the joint id
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id == -1:
        raise ValueError(f"Joint '{joint_name}' not found.")

    # Get the address of the joint's qpos in the qpos array
    qpos_addr = model.jnt_qposadr[joint_id]

    # Get the qpos value
    return data.qpos[qpos_addr]

def get_joint_id_from_name(model, name: str) -> int:
    """Return the id of a specified joint"""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def get_joint_addr_from_name(model, name: str) -> int:
    """Return the address of a specified joint"""
    return model.joint(name).qposadr


def get_actuator_names(model):
    actuator_names = [model.actuator(k).name for k in range(0, model.nu)]
    return actuator_names
