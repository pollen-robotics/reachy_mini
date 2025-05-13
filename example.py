import time
from stewart_little_control.stewart_little_control.control import StewartLittleControl
import numpy as np
from scipy.spatial.transform import Rotation as R

control = StewartLittleControl()

s = time.time()
while True:
    t = time.time() - s
    pose = np.eye(4)
    euler_rot = [
        0,
        0,
        # 0.2 * np.sin(2 * np.pi * 0.5 * t),
        0.2 * np.sin(2 * np.pi * 0.5 * t + np.pi),
    ]
    rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
    pose[:3, :3] = rot_mat

    control.set_pose(pose)
    time.sleep(0.01)
