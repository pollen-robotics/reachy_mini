"""Module for defining and playing motion moves on the ReachyMini robot.

This module provides the base class for moves, allowing for the creation of custom motions and the ability to play them on the robot.

"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from reachy_mini import ReachyMini
    from reachy_mini.daemon.backend.abstract import Backend


from reachy_mini.utils.interpolation import distance_between_poses


class Move(ABC):
    """Abstract base class for defining a move on the ReachyMini robot."""

    ms_per_degree = 10
    ms_per_magic_mm = 10

    @property
    @abstractmethod
    def duration(self) -> float:
        """Duration of the move in seconds."""
        pass

    @abstractmethod
    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Evaluate the move at time t.

        Returns:
            head: The head position (4x4 homogeneous matrix).
            antennas: The antennas positions (rad).
            body_yaw: The body yaw angle (rad).

        """

    def play_on(
        self,
        reachy_mini: "Backend | ReachyMini",
        repeat: int = 1,
        frequency: float = 100.0,
        start_goto: bool = False,
        is_relative: bool = False,
    ):
        """Play the move on the ReachyMini robot.

        Args:
            reachy_mini: The ReachyMini instance to control.
            repeat: Number of times to repeat the move.
            frequency: Frequency of updates in Hz.
            start_goto: Whether to interpolate to the starting position before playing the move.
            is_relative: If True, treat move as relative offsets.

        """
        asyncio.run(
            self.async_play_on(
                reachy_mini=reachy_mini,
                repeat=repeat,
                frequency=frequency,
                start_goto=start_goto,
                is_relative=is_relative,
            )
        )

    async def async_play_on(
        self,
        reachy_mini: "Backend | ReachyMini",
        repeat: int = 1,
        frequency: float = 100.0,
        start_goto: bool = False,
        is_relative: bool = False,
    ):
        """Play asynchronously the move on the ReachyMini robot.

        Args:
            reachy_mini: The ReachyMini instance to control.
            repeat: Number of times to repeat the move.
            frequency: Frequency of updates in Hz.
            start_goto: Whether to interpolate to the starting position before playing the move.
            is_relative: If True, treat move as relative offsets.

        """
        dt = 1.0 / frequency

        if start_goto:
            # Interpolation phase to reach the first target pose.
            start_head_pose, start_antennas_positions, start_body_yaw = self.evaluate(0)

            _, cur_antenna_joints = reachy_mini.get_present_antenna_joint_positions()
            current_head_pose = reachy_mini.get_current_head_pose()
            _, _, distance_to_goal = distance_between_poses(
                np.array(start_head_pose),
                current_head_pose,
            )
            head_interpol_duration = distance_to_goal * self.ms_per_magic_mm / 1000

            antenna_dist = max(
                abs(cur_antenna_joints - np.array(start_antennas_positions))
            )
            antenna_dist = np.rad2deg(antenna_dist)
            antenna_interpol_duration = antenna_dist * self.ms_per_degree / 1000

            first_duration = max(head_interpol_duration, antenna_interpol_duration)
            reachy_mini.goto_target(
                start_head_pose,
                start_antennas_positions,
                body_yaw=start_body_yaw,
                duration=first_duration,
                method="minjerk",
            )

        for _ in range(repeat):
            t0 = time.time()

            while True:
                t = time.time() - t0

                if t > self.duration:
                    break

                head, antennas, body_yaw = self.evaluate(t)

                reachy_mini.set_target(
                    head=head,
                    antennas=antennas,
                    body_yaw=body_yaw,
                    is_relative=is_relative,
                )

                end = time.time() - t0
                loop_duration = end - t
                sleep_duration = max(0, dt - loop_duration)
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)
