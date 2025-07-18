import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as reachy:
    # Move the head and antennas to a specific position
    reachy.goto_target(
        head=create_head_pose(y=-10, mm=True),
        antennas=np.deg2rad([45, 45]),
        duration=2.0,
        body_yaw=np.deg2rad(-30),
    )

    reachy.goto_target(
        head=create_head_pose(y=10, mm=True),
        antennas=np.deg2rad([45, 45]),
        duration=2.0,
        body_yaw=np.deg2rad(30),
    )
