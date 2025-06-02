from stewart_little_control.io import Client
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from stewart_little_control.command import ReachyMiniCommand

client = Client()

while True:
    pose = np.eye(4)
    euler_rot = [
        0,
        0,
        1.0 * np.sin(2 * np.pi * 0.5 * time.time() + np.pi),
    ]
    rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
    pose[:3, :3] = rot_mat
    pose[:3, 3][2] += 0.01 * np.sin(2 * np.pi * 0.5 * time.time())
    antennas = np.array([1, 1]) * np.sin(2 * np.pi * 1.5 * time.time())

    client.send_command(
        ReachyMiniCommand(
            head_pose=pose,
            antennas_orientation=antennas,
            offset_zero=True,
        )
    )
    time.sleep(0.02)
