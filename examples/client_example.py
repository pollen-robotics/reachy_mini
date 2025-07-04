"""Reachy Mini Client Example."""

import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini

with ReachyMini(spawn_daemon=True, use_sim=False) as reachy_mini:
    try:
        while True:
            pose = np.eye(4)
            pose[:3, 3][2] = 0.005 * np.sin(2 * np.pi * 0.3 * time.time() + np.pi)
            euler_rot = [
                0,
                0,
                0.5 * np.sin(2 * np.pi * 0.3 * time.time() + np.pi),
            ]
            rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
            pose[:3, :3] = rot_mat
            pose[:3, 3][2] += 0.01 * np.sin(2 * np.pi * 0.5 * time.time())
            antennas = np.array([1, 1]) * np.sin(2 * np.pi * 0.5 * time.time())

            reachy_mini.set_target(head=pose, antennas=antennas)

            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
