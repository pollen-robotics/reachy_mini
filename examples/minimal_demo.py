"""Minimal demo for Reachy Mini."""

import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini(media_backend="no_media") as mini:
    try:
        while True:
            t = time.time()
            x, y, z = 0.0, 0.0, 0.0
            roll, pitch, yaw = 0.0, 0.0, 0.0

            antennas_offset = np.deg2rad(20 * np.sin(2 * np.pi * 0.5 * time.time()))
            pitch = np.deg2rad(10 * np.sin(2 * np.pi * 0.5 * time.time()))

            head_pose = create_head_pose(
                x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, degrees=False, mm=False
            )
            mini.set_target(head=head_pose, antennas=(antennas_offset, antennas_offset))
    except KeyboardInterrupt:
        print("\nCtrl-C received. Shutting down...")
