"""Minimal demo for Reachy Mini."""

# START doc_example

import time

import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini(media_backend="no_media") as mini:
    mini.goto_target(create_head_pose(), antennas=[0.0, 0.0], duration=1.0)
    try:
        while True:
            t = time.time()

            antennas_offset = np.deg2rad(20 * np.sin(2 * np.pi * 0.5 * t))
            pitch = np.deg2rad(10 * np.sin(2 * np.pi * 0.5 * t))

            head_pose = create_head_pose(
                roll=0.0,
                pitch=pitch,
                yaw=0.0,
                degrees=False,
                mm=False,
            )
            mini.set_target(head=head_pose, antennas=(antennas_offset, antennas_offset))
    except KeyboardInterrupt:
        pass

# END doc_example
