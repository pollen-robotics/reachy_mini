import mujoco
import mujoco.viewer
import time
import os
from pathlib import Path

# from stewart_little_control.mujoco_utils import get_joint_qpos
from stewart_little_control import IKWrapper
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


ik_wrapper = IKWrapper()

pose = np.eye(4)
angles_rad = ik_wrapper.ik(pose)
sign = [-1, 1, 1, -1, 1, -1]
offset = (
    np.deg2rad(-22.5) - 0.076
)  # where does this 0.076 come from ?? Found empirically
step = 0
t = 0
all_start_t = time.time()
while True:
    start_t = time.time()
    t = time.time() - all_start_t

    if step % decimation != 0:
        # Control here

        pose = np.eye(4)
        pose[:3, 3][2] += 0.02 * np.sin(2 * np.pi * 0.5 * t)
        angles_rad = ik_wrapper.ik(pose)

        # print(angles_rad + offset)
        data.ctrl[:] = (angles_rad + offset) * sign
        pass

    mujoco.mj_step(model, data)
    viewer.sync()

    took = time.time() - start_t
    time.sleep(max(0, model.opt.timestep - took))
    step += 1
