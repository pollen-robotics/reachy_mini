from stewart_little_control import PlacoIK
import mujoco
import mujoco.viewer
import time
import os
from pathlib import Path
from stewart_little_control.mujoco_utils import get_joint_qpos
import numpy as np
from stewart_little_control.mujoco_utils import (
    get_homogeneous_matrix_from_euler,
    get_joints,
)

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent
NEUTRAL_POSITION = np.array([0.0, 0.0, 0.177 - 0.0075])
NEUTRAL_EULER_ANGLES = np.array([0.0, 0.0, 0.0]) # roll, pitch, yaw (radians)

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

init_pose = get_homogeneous_matrix_from_euler(
                position=NEUTRAL_POSITION,
                euler_angles=NEUTRAL_EULER_ANGLES,
            )
init_pose[:3, 3][2] = 0.177 - 0.0075
angles_rad = placo_ik.ik(init_pose)
step = 0
t = 0
all_start_t = time.time()
while True:
    start_t = time.time()
    t = time.time() - all_start_t

    if step % decimation != 0:
        # Control here

        current_pose_params = {
            "position": NEUTRAL_POSITION.copy(),
            "euler_angles": NEUTRAL_EULER_ANGLES.copy(),
        }
        val = 0.05 * np.sin(2 * np.pi * 0.5 * t)
        ang = np.deg2rad(30) * np.sin(2 * np.pi * 0.5 * t)

        # current_pose_params["position"][0] += val
        current_pose_params["euler_angles"][2] += ang
        pose = get_homogeneous_matrix_from_euler(
                position=tuple(current_pose_params["position"]),
                euler_angles=tuple(current_pose_params["euler_angles"]),
            )
        print(val)
        # pose[:3, 3][2] -= 0.02
        angles_rad = placo_ik.ik(pose)

        data.ctrl[:] = angles_rad

    mujoco.mj_step(model, data)
    viewer.sync()

    took = time.time() - start_t
    time.sleep(max(0, model.opt.timestep - took))
    step += 1
