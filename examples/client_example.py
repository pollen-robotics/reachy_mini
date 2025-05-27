from stewart_little_control import Client
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

client = Client()
while True:
    # sleep_pose = np.array(
    #     [
    #         [0.827, -0.005, 0.562, -0.032],
    #         [0.02, 1.0, -0.019, 0.008],
    #         [-0.562, 0.027, 0.827, 0.129],
    #         [0.0, 0.0, 0.0, 1.0],
    #     ]
    # )
    # client.send_pose(sleep_pose, antennas=[0, 0], offset_zero=False)
    # time.sleep(0.02)
    # continue
    pose = np.eye(4)
    pose[:3, 3][2] = 0.177  # Set the height of the head
    euler_rot = [
        # 0.3 * np.sin(2 * np.pi * 0.3 * time.time() + np.pi),
        0,
        0,
        # 0
        1.0 * np.sin(2 * np.pi * 0.3 * time.time() + np.pi),
    ]
    rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
    pose[:3, :3] = rot_mat
    pose[:3, 3][2] += 0.01 * np.sin(2 * np.pi * 1 * time.time())
    # pose[:3, 3][1] += 0.01 * np.sin(2 * np.pi * 0.5 * time.time())
    # # antennas = np.array([0.5, 0.5]) * np.sin(2 * np.pi * 1.0 * time.time())
    antennas=[0,0]
    client.send_pose(pose, antennas=antennas, offset_zero=False)
    time.sleep(0.02)
