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

def get_joints(model, data):
    joints = []
    joints_names =  ["1", "2", "3", "4", "5", "left_antenna", "right_antenna", "6"]
    for i in range(8):
        joints.append(get_joint_qpos(model, data, joints_names[i]))
    return joints



# def get_joints(model, data):
#     """
#     Gets qpos for joints named "1" through "6".
#     Returns a 1D NumPy array.
#     """
#     num_joints_to_get = 6
#     joint_qpos_values = np.empty(num_joints_to_get, dtype=float)
#     for i in range(num_joints_to_get):
#         joint_name = str(i + 1)
#         joint_qpos_values[i] = get_joint_qpos(model, data, joint_name)
#     return joint_qpos_values