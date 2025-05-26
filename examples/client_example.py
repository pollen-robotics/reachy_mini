from stewart_little_control import Client
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

client = Client()
while True:
    pose = np.eye(4)
    euler_rot = [
        0,
        0,
        1.0 * np.sin(2 * np.pi * 0.2 * time.time() + np.pi),
    ]
    rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
    pose[:3, :3] = rot_mat
    pose[:3, 3][2] += 0.01 * np.sin(2 * np.pi * 0.2 * time.time())
    antennas = np.array([0.5, 0.5]) * np.sin(2 * np.pi * 1.0 * time.time())
    client.send_pose(pose, antennas=antennas, offset_zero=True)
    time.sleep(0.02)
