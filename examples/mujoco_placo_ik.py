from stewart_little_control import PlacoIK
import mujoco
import mujoco.viewer
import time
import os
from pathlib import Path
from stewart_little_control.mujoco_utils import get_joint_qpos
import numpy as np

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

placo_ik = PlacoIK(f"{ROOT_PATH}/descriptions/stewart_little_magnet/")

init_pose = np.eye(4)
init_pose[:3, 3][2] = 0.155
angles_rad = placo_ik.ik(init_pose)
step = 0
t = 0
all_start_t = time.time()
while True:
    start_t = time.time()
    t = time.time() - all_start_t

    if step % decimation != 0:
        # Control here

        pose = init_pose.copy()
        pose[:3, 3][0] += 0.01 * np.sin(2 * np.pi * 0.5 * t)
        # pose[:3, 3][2] -= 0.02
        angles_rad = placo_ik.ik(pose)
        # print("angles deg", np.rad2deg(angles_rad))
        print(pose[:3, 3][0])

        data.ctrl[:] = angles_rad

    mujoco.mj_step(model, data)
    viewer.sync()

    took = time.time() - start_t
    time.sleep(max(0, model.opt.timestep - took))
    step += 1
