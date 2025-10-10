"""Reachy Mini Motion Sequence Example."""

import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini

with ReachyMini(media_backend="no_media") as reachy_mini:
    reachy_mini.goto_target(np.eye(4), antennas=[0.0, 0.0], duration=1.0)
    try:
        while True:
            pose = np.eye(4)

            t = 0
            t0 = time.time()
            s = time.time()
            while time.time() - s < 2.0:
                t = time.time() - t0
                euler_rot = np.array([0, 0.0, 0.7 * np.sin(2 * np.pi * 0.5 * t)])
                rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
                pose[:3, :3] = rot_mat
                reachy_mini.set_target(head=pose, antennas=[0, 0])
                time.sleep(0.01)

            s = time.time()
            while time.time() - s < 2.0:
                t = time.time() - t0
                euler_rot = np.array([0, 0.3 * np.sin(2 * np.pi * 0.5 * t), 0])
                rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
                pose[:3, :3] = rot_mat
                reachy_mini.set_target(head=pose, antennas=[0, 0])
                time.sleep(0.01)

            s = time.time()
            while time.time() - s < 2.0:
                t = time.time() - t0
                euler_rot = np.array([0.3 * np.sin(2 * np.pi * 0.5 * t), 0, 0])
                rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
                pose[:3, :3] = rot_mat
                reachy_mini.set_target(head=pose, antennas=[0, 0])
                time.sleep(0.01)

            s = time.time()
            while time.time() - s < 2.0:
                t = time.time() - t0
                pose = np.eye(4)
                pose[:3, 3][2] += 0.025 * np.sin(2 * np.pi * 0.5 * t)
                reachy_mini.set_target(head=pose, antennas=[0, 0])
                time.sleep(0.01)

            s = time.time()
            while time.time() - s < 2.0:
                t = time.time() - t0
                antennas = [
                    0.5 * np.sin(2 * np.pi * 0.5 * t),
                    -0.5 * np.sin(2 * np.pi * 0.5 * t),
                ]
                reachy_mini.set_target(head=pose, antennas=antennas)
                time.sleep(0.01)

            s = time.time()
            while time.time() - s < 5.0:
                t = time.time() - t0
                pose[:3, 3] = [
                    0.015 * np.sin(2 * np.pi * 1.0 * t),
                    0.015 * np.sin(2 * np.pi * 1.0 * t + np.pi / 2),
                    0.0,
                ]
                reachy_mini.set_target(head=pose, antennas=[0, 0])
                time.sleep(0.01)

            pose[:3, 3] = [0, 0, 0.0]
            reachy_mini.set_target(head=pose, antennas=[0, 0])

            time.sleep(0.5)

            pose[:3, 3] = [0.02, 0.02, 0.0]
            reachy_mini.set_target(head=pose, antennas=[0, 0])
            time.sleep(0.5)

            pose[:3, 3] = [0.00, 0.02, 0.0]
            euler_rot = np.array([0, 0, 0.5])
            rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
            pose[:3, :3] = rot_mat
            reachy_mini.set_target(head=pose, antennas=[0, 0])
            time.sleep(0.5)

            pose[:3, 3] = [0.00, -0.02, 0.0]
            euler_rot = np.array([0, 0, -0.5])
            rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
            pose[:3, :3] = rot_mat
            reachy_mini.set_target(head=pose, antennas=[0, 0])
            time.sleep(0.5)

            pose[:3, 3] = [0, 0, 0.0]
            reachy_mini.set_target(head=pose, antennas=[0, 0])
            time.sleep(2)

    except KeyboardInterrupt:
        pass
