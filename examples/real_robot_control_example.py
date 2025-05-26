import time
from stewart_little_control import StewartLittleControl
import numpy as np
from scipy.spatial.transform import Rotation as R

control = StewartLittleControl()

s = time.time()
while True:
    t = time.time() - s
    pose = np.eye(4)
    pose[:3, 3][2] = 0.177
    # pose[:3, 3][0] += 0.015 * np.sin(2 * np.pi * 1.0 * t)
    # pose[:3, 3][1] += 0.015 * np.sin(2 * np.pi * 1.0 * t)
    # pose[:3, 3][2] += 0.015 * np.sin(2 * np.pi * 1.0 * t)
    # print(pose[:3, 3][2])
    # pose[:3, 3][2] -= 0.02
    euler_rot = [
        0.1 * np.sin(2 * np.pi * 4.0 * t + np.pi),
        0,
        0
        # 0, 0,
        # 0.3 * np.sin(2 * np.pi * 0.5 * t + np.pi),
        # 0.3 * np.sin(2 * np.pi * 0.5 * t + np.pi),
        # 0,
        # 0.2 * np.sin(2 * np.pi * 0.5 * t),
        # 0.5 * np.sin(2 * np.pi * 0.5 * t + np.pi),
    ]
    rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
    pose[:3, :3] = rot_mat
    print(pose[:3, 3][0])


    control.set_pose(pose)
    time.sleep(0.01)
