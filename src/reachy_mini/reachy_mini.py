import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import numpy as np
import pygame
from scipy.spatial.transform import Rotation as R

import reachy_mini.led_animations
from reachy_mini.io import Client, NeoPixelRing
from reachy_mini.placo_kinematics import PlacoKinematics
from reachy_mini.utils import (
    daemon_check,
    minimum_jerk,
    time_trajectory,
    linear_pose_interpolation,
)

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent

try:
    pygame.mixer.init()
except pygame.error as e:
    print(f"Failed to initialize pygame mixer: {e}")
    pygame.mixer = None

# Behavior definitions
INIT_HEAD_POSE = np.eye(4)

SLEEP_HEAD_JOINT_POSITIONS = [
    0,
    -0.9848156658225817,
    1.2624661884298831,
    -0.24390294527381684,
    0.20555342557667577,
    -1.2363885150358267,
    1.0032234352772091,
]


SLEEP_ANTENNAS_JOINT_POSITIONS = [3.05, -3.05]
SLEEP_HEAD_POSE = np.array(
    [
        [0.911, 0.004, 0.413, -0.021],
        [-0.004, 1.0, -0.001, 0.001],
        [-0.413, -0.001, 0.911, -0.044],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


class ReachyMini:
    def __init__(
        self,
        localhost_only: bool = True,
        spawn_daemon: bool = False,
        use_sim: bool = True,
    ) -> None:
        daemon_check(spawn_daemon, use_sim)
        self.client = Client(localhost_only)

        self.client.wait_for_connection()
        self._last_head_pose = None

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
            assert antennas.shape == (2,), (
                "Antennas must be a 1D array with two elements."
            )
            antenna_joint_positions = antennas.tolist()
        else:
            antenna_joint_positions = None

        self._send_joint_command(head_joint_positions, antenna_joint_positions)
        self._last_head_pose = head

    def goto_position(
        self,
        head: Optional[np.ndarray] = None,  # 4x4 pose matrix
        antennas: Optional[np.ndarray] = None,  # [left_angle, right_angle] (in rads)
        duration: float = 0.5,  # Duration in seconds for the movement
        method="default",
    ):
        self._goto_task_positions(
            target_head_pose=head,
            antennas_joint_positions=antennas,
            duration=duration,
            method=method,
        )

    def set_torque(self, on: bool):
        """
        Set the torque state of the motors.
        :param on: If True, enables torque; if False, disables it.
        """
        self.client.send_command(json.dumps({"torque": on}))

    def wake_up(self):

        # Lights up
        self.play_led_animation(reachy_mini.led_animations.wake_spiral)

        self.goto_position(INIT_HEAD_POSE, antennas=[0.0, 0.0], duration=2)
        time.sleep(0.1)

        # Toudoum
        self.play_sound("proud2.wav")

        self.clear_led()

        # Roll 20° to the left
        pose = INIT_HEAD_POSE.copy()
        pose[:3, :3] = R.from_euler("xyz", [20, 0, 0], degrees=True).as_matrix()
        self.goto_position(pose, duration=0.2)

        # Go back to the initial position
        self.goto_position(INIT_HEAD_POSE, duration=0.2)

    def goto_sleep(self):

        # Lights off
        self.play_led_animation(reachy_mini.led_animations.sleep_heartbeat)

        # Check if we are too far from the initial position
        # Move to the initial position if necessary
        current_positions, _ = self._get_current_joint_positions()
        init_positions = self.head_kinematics.ik(INIT_HEAD_POSE)
        dist = np.linalg.norm(np.array(current_positions) - np.array(init_positions))
        if dist > 0.2:
            self.goto_position(INIT_HEAD_POSE, antennas=[0.0, 0.0], duration=1)
            time.sleep(0.2)

        # Move to the sleep position
        self._goto_joint_positions(
            head_joint_positions=SLEEP_HEAD_JOINT_POSITIONS,
            antennas_joint_positions=SLEEP_ANTENNAS_JOINT_POSITIONS,
            duration=2,
        )
        self._last_head_pose = SLEEP_HEAD_POSE

    # Multimedia methods
    def play_sound(self, sound_file: str):
        if pygame.mixer is None:
            print("Pygame mixer is not initialized. Cannot play sound.")
            return
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
            assert len(head_joint_positions) == 7, (
                f"Head joint positions must have length 7, got {head_joint_positions}."
            )
            cmd["head_joint_positions"] = list(head_joint_positions)

        if antennas_joint_positions is not None:
            assert len(antennas_joint_positions) == 2, "Antennas must have length 2."
            cmd["antennas_joint_positions"] = list(antennas_joint_positions)

        if not cmd:
            raise ValueError(
                "At least one of head_joint_positions or antennas must be provided."
            )
        self.client.send_command(json.dumps(cmd))

    def _goto_task_positions(
        self,
        target_head_pose,
        antennas_joint_positions: Optional[List[float]] = None,
        duration: float = 0.5,
        method="default",
    ):
        cur_head_joints, cur_antennas_joints = self._get_current_joint_positions()

        if self._last_head_pose is None:
            start_head_pose = self.head_kinematics.fk(cur_head_joints)
        else:
            start_head_pose = self._last_head_pose

        target_head_pose = (
            self.head_kinematics.fk(cur_head_joints)
            if target_head_pose is None
            else target_head_pose
        )

        start_antennas = np.array(cur_antennas_joints)
        target_antennas = (
            start_antennas
            if antennas_joint_positions is None
            else np.array(antennas_joint_positions)
        )

        t0 = time.time()
        while time.time() - t0 < duration:
            t = time.time() - t0

            interp_time = time_trajectory(t / duration, method=method)
            interp_head_pose = linear_pose_interpolation(
                start_head_pose, target_head_pose, interp_time
            )
            interp_antennas_joint = (
                start_antennas + (target_antennas - start_antennas) * interp_time
            )

            self.set_position(interp_head_pose, interp_antennas_joint)

            time.sleep(0.01)

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

    def set_led_colors(
        self,
        colors: Union[
            List[Optional[Tuple[int, int, int]]], Dict[int, Tuple[int, int, int]]
        ],
        duration: Optional[float] = None,
    ):
        """
        Set the colors of the LEDs.
        Args:
            colors: Either a list of (r,g,b) tuples with None for unchanged LEDs,
                    or a dict with LED index as key and (r,g,b) tuple as value
        """
        cmd = {
            "set_led_colors": (colors, duration),
        }

        self.client.send_command(json.dumps(cmd))

    def clear_led(self):
        """
        Clear the LED colors.
        This will set all LEDs to off.
        """
        cmd = {
            "clear_led": None,
        }

        self.client.send_command(json.dumps(cmd))

    def play_led_animation(self, animation):
        """
        Play a LED animation.
        Args:
            animation: the animation array, which is a list of tuples
                       where each tuple contains RGB values for each LED and duration.
        """

        for step in animation:
            self.set_led_colors(*step)
