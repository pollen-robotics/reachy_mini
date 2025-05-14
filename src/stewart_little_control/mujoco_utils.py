import mujoco

def get_joint_qpos(model, data, joint_name):
    # Get the joint id
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

    # Get the address of the joint's qpos in the qpos array
    qpos_addr = model.jnt_qposadr[joint_id]

    # Get the qpos value
    joint_qpos = data.qpos[qpos_addr]
    return joint_qpos
