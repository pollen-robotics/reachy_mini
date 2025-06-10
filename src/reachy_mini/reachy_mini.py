import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from reachy_mini.io import Client
from reachy_mini.placo_kinematics import PlacoKinematics


ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent


class ReachyMini:
    def __init__(self) -> None:
        self.client = Client(localhost_only=False)
        self.head_kinematics = PlacoKinematics(
            f"{ROOT_PATH}/descriptions/reachy_mini/urdf/",
            sim=True,
        )

    def __enter__(self):
        self.set_torque(on=True)
        self.wake_up()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.goto_sleep()
        self.set_torque(on=False)

    def set_position(
        self,
        head: Optional[np.ndarray],  # 4x4 pose matrix
        antennas: Optional[np.ndarray] = None,  # [left_angle, right_angle] (in rads)
    ):
        """
        Set the position of the head and/or antennas.
        :param head: 4x4 pose matrix representing the head pose.
        :param antennas: Optional 1D array with two elements representing the angles of the antennas in radians.
        """
        if head is not None:
            assert head.shape == (4, 4), "Head pose must be a 4x4 matrix."
            head_joint_positions = self.head_kinematics.ik(head)
        else:
            head_joint_positions = None

        self._send_joint_command(
            head_joint_positions=head_joint_positions,
            antennas=antennas.tolist() if antennas is not None else None,
        )

    def goto_position(
        self,
        head: Optional[np.ndarray] = None,  # 4x4 pose matrix
        antennas: Optional[np.ndarray] = None,  # [left_angle, right_angle] (in rads)
        duration: float = 0.5,  # Duration in seconds for the movement
    ):
        raise NotImplementedError()

    def _goto_joint_positions(
        self,
        head_joint_positions: Optional[
            List[float]
        ] = None,  # [yaw, stewart_platform x 6] length 7
        antennas: Optional[List[float]] = None,  # [left_angle, right_angle] length 2
        duration: float = 0.5,  # Duration in seconds for the movement
    ):
        # """
        # Move the head and/or antennas to the specified joint positions.
        # :param head_joint_positions: List of joint positions for the head.
        # :param antennas: List of angles for the antennas.
        # :param duration: Duration in seconds for the movement.
        # """
        # self._send_joint_command(head_joint_positions, antennas)
        pass

    def set_torque(self, on: bool):
        """
        Set the torque state of the motors.
        :param on: If True, enables torque; if False, disables it.
        """
        self.client.send_command(json.dumps({"torque": on}))

    # Behavior definitions
    def wake_up(self):
        pass

    def goto_sleep(self):
        pass

    # Low-level joints methods

    def _get_current_joint_positions(self) -> List[float]:
        raise NotImplementedError()

    def _send_joint_command(
        self,
        head_joint_positions: Optional[
            List[float]
        ],  # [yaw, stewart_platform x 6] length 7
        antennas: Optional[List[float]],  # [left_angle, right_angle] length 2
    ):
        cmd = {}

        if head_joint_positions is not None:
            assert len(head_joint_positions) == 7, (
                f"Head joint positions must have length 7, got {head_joint_positions}."
            )
            cmd["head_joint_positions"] = head_joint_positions

        if antennas is not None:
            assert len(antennas) == 2, "Antennas must have length 2."
            cmd["antennas"] = antennas

        if not cmd:
            raise ValueError(
                "At least one of head_joint_positions or antennas must be provided."
            )
        self.client.send_command(json.dumps(cmd))
