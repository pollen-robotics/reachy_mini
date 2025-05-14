import mujoco
import mujoco.viewer
import time
import os
from pathlib import Path
from stewart_little_control.mujoco_utils import get_joint_qpos, get_joints, get_homogeneous_matrix_from_euler
import numpy as np
from stewart_little_control import IKWrapper




ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent

model = mujoco.MjModel.from_xml_path(
    f"{ROOT_PATH}/descriptions/stewart_little_magnet/scene.xml"
)
data = mujoco.MjData(model)
model.opt.timestep = 0.002  # s, simulation timestep, 500hz
decimation = 10  # -> 50hz control loop

viewer = mujoco.viewer.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False
)

ik_wrapper = IKWrapper()

step = 0
while True:
    start_t = time.time()

    # print(get_joint_qpos(model, data, "1"))
    print(get_joints(model, data))
    x = 0.02*np.sin(2 * np.pi * time.time())
    val = np.deg2rad(10)*np.sin(2 * np.pi * time.time())
    
    pose = get_homogeneous_matrix_from_euler(
        position=(0.0, 0.0, 0.0),
        euler_angles=(val, 0.0, 0.0),
        degrees=False,
    )
    angles_rad = ik_wrapper.ik(pose)
    print(f"angles_rad: {angles_rad}")
    
    if step % decimation != 0:
        # Control here
        # data.ctrl[:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
        data.ctrl[:] = angles_rad
        
        pass

    mujoco.mj_step(model, data)
    viewer.sync()

    took = time.time() - start_t
    time.sleep(max(0, model.opt.timestep - took))
    step += 1
