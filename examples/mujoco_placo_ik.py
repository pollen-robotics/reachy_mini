from reachy_mini import PlacoKinematics
import mujoco
import mujoco.viewer
import time
import os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent

model = mujoco.MjModel.from_xml_path(
    f"{ROOT_PATH}/descriptions/reachy_mini/mjcf/scene.xml"
)
data = mujoco.MjData(model)
model.opt.timestep = 0.002  # s, simulation timestep, 500hz
decimation = 10  # -> 50hz control loop

viewer = mujoco.viewer.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False
)

placo_kinematics = PlacoKinematics(
    f"{ROOT_PATH}/descriptions/reachy_mini/urdf/",
)

init_pose = np.eye(4)
init_pose[:3, 3][2] = 0.177
angles_rad = placo_kinematics.ik(init_pose)
step = 0
t = 0
all_start_t = time.time()
while True:
    start_t = time.time()
    t = time.time() - all_start_t

    if step % decimation != 0:
        # Control here

        pose = init_pose.copy()
        pose[:3, 3][2] += 0.02 * np.sin(2 * np.pi * 0.5 * t)
        euler_rot = [
            0,
            0.2 * np.sin(2 * np.pi * 0.25 * t + np.pi),
            1.0 * np.sin(2 * np.pi * 0.5 * t + np.pi),
        ]
        rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
        pose[:3, :3] = rot_mat
        angles_rad = placo_kinematics.ik(pose)
        # fk = placo_kinematics.fk(angles_rad)
        # print("diff", np.linalg.norm(fk - pose))
        data.ctrl[:] = angles_rad

    mujoco.mj_step(model, data)
    viewer.sync()

    took = time.time() - start_t
    time.sleep(max(0, model.opt.timestep - took))
    step += 1
