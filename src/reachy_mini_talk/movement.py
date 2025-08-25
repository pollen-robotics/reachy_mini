import time
import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


class MovementManager:
    def __init__(self, current_robot: ReachyMini):
        self.current_robot = current_robot
        self.current_head_pose = np.eye(4)
        self.moving_start = time.monotonic()
        self.moving_for = 0.0
        self.is_head_tracking = False
        self.speech_head_offsets = [0.0] * 6

    def set_offsets(self, offsets: list[float]) -> None:
        """Used by AudioSync callback to update speech offsets"""
        self.speech_head_offsets = list(offsets)

    def set_neutral(self) -> None:
        self.speech_head_offsets = [0.0] * 6
        self.current_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.current_robot.set_target(head=self.current_head_pose, antennas=(0.0, 0.0))

    def reset_head_pose(self) -> None:
        self.current_head_pose = np.eye(4)
