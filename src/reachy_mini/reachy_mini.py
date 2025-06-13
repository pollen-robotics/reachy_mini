import json
import os
import time
from pathlib import Path
from typing import List, Optional

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import numpy as np
import pygame
from scipy.spatial.transform import Rotation as R

from reachy_mini.io import Client
from reachy_mini.placo_kinematics import PlacoKinematics
from reachy_mini.utils import daemon_check, minimum_jerk

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent

pygame.mixer.init()


class ReachyMini:
    def __init__(self, localhost_only: bool = True, spawn_daemon: bool = False, use_sim: bool = True) -> None:
        daemon_check(spawn_daemon, use_sim)
        self.client = Client(localhost_only)
        self.client.wait_for_connection()

        self.head_kinematics = PlacoKinematics(
            f"{ROOT_PATH}/descriptions/reachy_mini/urdf/",
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.client.disconnect()

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

        if antennas is not None:
            assert antennas.shape == (
                2,
            ), "Antennas must be a 1D array with two elements."
            antenna_joint_positions = antennas.tolist()
        else:
            antenna_joint_positions = None

        self._send_joint_command(head_joint_positions, antenna_joint_positions)

    def goto_position(
        self,
        head: Optional[np.ndarray] = None,  # 4x4 pose matrix
        antennas: Optional[np.ndarray] = None,  # [left_angle, right_angle] (in rads)
        duration: float = 0.5,  # Duration in seconds for the movement
    ):
        head_joint_positions = None
        if head is not None:
            head_joint_positions = self.head_kinematics.ik(head)

        self._goto_joint_positions(
            head_joint_positions=head_joint_positions,
            antennas_joint_positions=antennas,
            duration=duration,
        )

    def set_torque(self, on: bool):
        """
        Set the torque state of the motors.
        :param on: If True, enables torque; if False, disables it.
        """
        self.client.send_command(json.dumps({"torque": on}))

    # Behavior definitions
    init_head_pose = np.eye(4)
    init_head_pose[2, 3] = 0.177

    sleep_head_joint_positions = [
        0.0,
        0.849,
        -1.292,
        0.472,
        0.047,
        1.31,
        -0.876,
    ]
    sleep_antennas_joint_positions = [3.05, -3.05]

    def wake_up(self):
        self.goto_position(self.init_head_pose, antennas=[0.0, 0.0], duration=2)
        time.sleep(0.1)

        # Toudoum
        self.play_sound("proud2.wav")

        # Roll 20° to the left
        pose = self.init_head_pose.copy()
        pose[:3, :3] = R.from_euler("xyz", [20, 0, 0], degrees=True).as_matrix()
        self.goto_position(pose, duration=0.2)

        # Go back to the initial position
        self.goto_position(self.init_head_pose, duration=0.2)

    def goto_sleep(self):
        # Check if we are too far from the initial position
        # Move to the initial position if necessary
        current_positions, _ = self._get_current_joint_positions()
        init_positions = self.head_kinematics.ik(self.init_head_pose)
        dist = np.linalg.norm(np.array(current_positions) - np.array(init_positions))
        if dist > 0.2:
            self.goto_position(self.init_head_pose, antennas=[0.0, 0.0], duration=1)
            time.sleep(0.2)

        # Move to the sleep position
        self._goto_joint_positions(
            head_joint_positions=self.sleep_head_joint_positions,
            antennas_joint_positions=self.sleep_antennas_joint_positions,
            duration=2,
        )

    # Multimedia methods
    def play_sound(self, sound_file: str):
        pygame.mixer.music.load(f"{ROOT_PATH}/src/assets/{sound_file}")
        pygame.mixer.music.play()

    # Low-level joints methods
    def _get_current_joint_positions(self) -> tuple[list[float], list[float]]:
        return self.client.get_current_joints()

    def _send_joint_command(
        self,
        head_joint_positions: Optional[
            List[float]
        ],  # [yaw, stewart_platform x 6] length 7
        antennas_joint_positions: Optional[
            List[float]
        ],  # [left_angle, right_angle] length 2
    ):
        cmd = {}

        if head_joint_positions is not None:
            assert (
                len(head_joint_positions) == 7
            ), f"Head joint positions must have length 7, got {head_joint_positions}."
            cmd["head_joint_positions"] = list(head_joint_positions)

        if antennas_joint_positions is not None:
            assert len(antennas_joint_positions) == 2, "Antennas must have length 2."
            cmd["antennas_joint_positions"] = list(antennas_joint_positions)

        if not cmd:
            raise ValueError(
                "At least one of head_joint_positions or antennas must be provided."
            )
        self.client.send_command(json.dumps(cmd))

    def _goto_joint_positions(
        self,
        head_joint_positions: Optional[
            List[float]
        ] = None,  # [yaw, stewart_platform x 6] length 7
        antennas_joint_positions: Optional[
            List[float]
        ] = None,  # [left_angle, right_angle] length 2
        duration: float = 0.5,  # Duration in seconds for the movement
    ):
        cur_head, cur_antennas = self._get_current_joint_positions()
        current = cur_head + cur_antennas

        target = []
        if head_joint_positions is not None:
            target.extend(head_joint_positions)
        else:
            target.extend(cur_head)
        if antennas_joint_positions is not None:
            target.extend(antennas_joint_positions)
        else:
            target.extend(cur_antennas)

        current = np.array(current)
        target = np.array(target)
        traj = minimum_jerk(current, target, duration)

        t0 = time.time()
        while time.time() - t0 < duration:
            t = time.time() - t0
            angles = traj(t)

            head_joint = angles[:7]  # First 7 angles for the head
            antennas_joint = angles[7:]

            self._send_joint_command(head_joint, antennas_joint)
            time.sleep(0.01)
