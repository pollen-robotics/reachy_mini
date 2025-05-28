from stewart_little_control import Client
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

client = Client()

while True:
    pose = np.eye(4)
    pose[:3, 3][2] = 0.177  # Set the height of the head
    pose[:3, 3][2] += 0.01 * np.sin(2 * np.pi * 0.3 * time.time() + np.pi)
    euler_rot = [
        0,
        0,
        1.0 * np.sin(2 * np.pi * 0.3 * time.time() + np.pi),
    ]
    rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
    pose[:3, :3] = rot_mat
    antennas = [
        np.sin(2 * np.pi * 1.0 * time.time()),
        np.sin(2 * np.pi * 1.0 * time.time()),
    ]
    client.send_pose(pose, antennas=antennas, offset_zero=False)
    time.sleep(0.02)
