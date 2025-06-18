import mujoco
import mujoco.viewer
import time
import os
from pathlib import Path
from reachy_mini.mujoco_utils import get_joint_qpos

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent

model = mujoco.MjModel.from_xml_path(
    f"{ROOT_PATH}/descriptions/reachy_mini/mjcf/scenes/empty.xml"
)
data = mujoco.MjData(model)
model.opt.timestep = 0.002  # s, simulation timestep, 500hz
decimation = 10  # -> 50hz control loop

viewer = mujoco.viewer.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False
)

step = 0
while True:
    start_t = time.time()

    print(get_joint_qpos(model, data, "1"))

    if step % decimation != 0:
        # Control here
        # data.ctrl[:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        pass

    mujoco.mj_step(model, data)
    viewer.sync()

    took = time.time() - start_t
    time.sleep(max(0, model.opt.timestep - took))
    step += 1
