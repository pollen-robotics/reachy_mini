"""Reachy Mini class for controlling a simulated or real Reachy Mini robot.

This class provides methods to control the head and antennas of the Reachy Mini robot,
set their target positions, and perform various behaviors such as waking up and going to sleep.

It also includes methods for multimedia interactions like playing sounds and looking at specific points in the image frame or world coordinates.
"""

import json
import os
import time
from typing import List, Optional, Union

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from importlib.resources import files

import cv2
import numpy as np
import pygame
from scipy.spatial.transform import Rotation as R

import reachy_mini
from reachy_mini.daemon.utils import daemon_check
from reachy_mini.io import Client
from reachy_mini.utils.interpolation import (
    linear_pose_interpolation,
    minimum_jerk,
    time_trajectory,
)

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

IMAGE_SIZE = (1280, 720)  # Width, Height in pixels


class ReachyMini:
    """Reachy Mini class for controlling a simulated or real Reachy Mini robot.

    Args:
        localhost_only (bool): If True, will only connect to localhost daemons, defaults to True.
        spawn_daemon (bool): If True, will spawn a daemon to control the robot, defaults to False.
        use_sim (bool): If True and spawn_daemon is True, will spawn a simulated robot, defaults to True.

    """

    urdf_root_path: str = str(
        files(reachy_mini).joinpath("descriptions/reachy_mini/urdf")
    )
    assets_root_path: str = str(files(reachy_mini).joinpath("assets/"))

    def __init__(
        self,
        localhost_only: bool = True,
        spawn_daemon: bool = False,
        use_sim: bool = True,
        timeout: float = 5.0,
        automatic_body_yaw: bool = False,
    ) -> None:
        """Initialize the Reachy Mini robot.

        Args:
            localhost_only (bool): If True, will only connect to localhost daemons, defaults to True.
            spawn_daemon (bool): If True, will spawn a daemon to control the robot, defaults to False.
            use_sim (bool): If True and spawn_daemon is True, will spawn a simulated robot, defaults to True.
            timeout (float): Timeout for the client connection, defaults to 5.0 seconds.
            automatic_body_yaw (bool): If True, the body yaw will be used to compute the IK and FK. Default is False.

        It will try to connect to the daemon, and if it fails, it will raise an exception.

        """
        daemon_check(spawn_daemon, use_sim)
        self.client = Client(localhost_only)
        self.client.wait_for_connection(timeout=timeout)
        self.set_automatic_body_yaw(automatic_body_yaw)
        self._last_head_pose = None

        self.K = np.array(
            [[550.3564, 0.0, 638.0112], [0.0, 549.1653, 364.589], [0.0, 0.0, 1.0]]
        )
        self.D = np.array([-0.0694, 0.1565, -0.0004, 0.0003, -0.0983])

        self.T_head_cam = np.eye(4)
        self.T_head_cam[:3, 3][:] = [0.0437, 0, 0.0512]
        self.T_head_cam[:3, :3] = np.array(
            [
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0],
            ]
        )

    def __enter__(self) -> "ReachyMini":
        """Context manager entry point for Reachy Mini."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Context manager exit point for Reachy Mini."""
        self.client.disconnect()

    def set_target(
        self,
        head: Optional[np.ndarray] = None,  # 4x4 pose matrix
        antennas: Optional[
            Union[np.ndarray, List[float]]
        ] = None,  # [left_angle, right_angle] (in rads)
        body_yaw: float = 0.0,  # Body yaw angle in radians
        check_collision: bool = False,  # Check for collisions before setting the position
    ) -> None:
        """Set the target pose of the head and/or the target position of the antennas.

        Args:
            head (Optional[np.ndarray]): 4x4 pose matrix representing the head pose.
            antennas (Optional[Union[np.ndarray, List[float]]]): 1D array with two elements representing the angles of the antennas in radians.
            body_yaw (Optional[float]): Body yaw angle in radians.
            check_collision (bool): If True, checks for collisions before setting the position. Beware that this will slow down the IK computation (~1ms).

        Raises:
            ValueError: If neither head nor antennas are provided, or if the shape of head is not (4, 4), or if antennas is not a 1D array with two elements.

        """
        if head is None and antennas is None:
            raise ValueError("At least one of head or antennas must be provided.")

        if head is not None and not head.shape == (4, 4):
            raise ValueError(f"Head pose must be a 4x4 matrix, got shape {head.shape}.")

        if antennas is not None and not len(antennas) == 2:
            raise ValueError(
                "Antennas must be a list or 1D np array with two elements."
            )

        self._set_check_collision(check_collision)
        self._set_joint_positions(None, list(antennas))
        self._set_head_pose(head, body_yaw)
        self._last_head_pose = head

    def goto_target(
        self,
        head: Optional[np.ndarray] = None,  # 4x4 pose matrix
        antennas: Optional[
            Union[np.ndarray, List[float]]
        ] = None,  # [left_angle, right_angle] (in rads)
        duration: float = 0.5,  # Duration in seconds for the movement, default is 0.5 seconds.
        method="default",  # can be "linear", "minjerk", "ease" or "cartoon", default is "default" (-> "minjerk" interpolation)
        body_yaw: float = 0.0,  # Body yaw angle in radians
        check_collision: bool = False,
    ):
        """Go to a target head pose and/or antennas position using task space interpolation, in "duration" seconds.

        Args:
            head (Optional[np.ndarray]): 4x4 pose matrix representing the target head pose.
            antennas (Optional[Union[np.ndarray, List[float]]]): 1D array with two elements representing the angles of the antennas in radians.
            duration (float): Duration of the movement in seconds.
            method (str): Interpolation method to use ("linear", "minjerk", "ease", "cartoon"). Default is "minjerk".
            body_yaw (float): Body yaw angle in radians.
            check_collision (bool): If True, checks for collisions before setting the position. Beware that this will slow down the IK computation (~1ms)!

        Raises:
            ValueError: If neither head nor antennas are provided, or if duration is not positive.

        """
        if head is None and antennas is None:
            raise ValueError("At least one of head or antennas must be provided.")

        if duration <= 0.0:
            raise ValueError(
                "Duration must be positive and non-zero. Use set_target() for immediate position setting."
            )

        cur_head_joints, cur_antennas_joints = self._get_current_joint_positions()
        start_body_yaw = cur_head_joints[0]
        cur_head_pose = self._get_current_head_pose()

        if self._last_head_pose is None:
            start_head_pose = cur_head_pose
        else:
            start_head_pose = self._last_head_pose

        target_head_pose = cur_head_pose if head is None else head

        start_antennas = np.array(cur_antennas_joints)
        target_antennas = start_antennas if antennas is None else np.array(antennas)

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
            interp_body_yaw_joint = (
                start_body_yaw + (body_yaw - start_body_yaw) * interp_time
            )

            self.set_target(
                interp_head_pose,
                list(interp_antennas_joint),
                body_yaw=interp_body_yaw_joint,
                check_collision=check_collision,
            )

            time.sleep(0.01)

    def wake_up(self) -> None:
        """Wake up the robot - go to the initial head position and play the wake up emote and sound."""
        self.goto_target(INIT_HEAD_POSE, antennas=[0.0, 0.0], duration=2)
        time.sleep(0.1)

        # Toudoum
        self.play_sound("proud2.wav")

        # Roll 20° to the left
        pose = INIT_HEAD_POSE.copy()
        pose[:3, :3] = R.from_euler("xyz", [20, 0, 0], degrees=True).as_matrix()
        self.goto_target(pose, duration=0.2)

        # Go back to the initial position
        self.goto_target(INIT_HEAD_POSE, duration=0.2)

    def goto_sleep(self) -> None:
        """Put the robot to sleep by moving the head and antennas to a predefined sleep position."""
        # Check if we are too far from the initial position
        # Move to the initial position if necessary
        current_positions, _ = self._get_current_joint_positions()
        # init_positions = self.head_kinematics.ik(INIT_HEAD_POSE)
        # Todo : get init position from the daemon?
        init_positions = [
            6.959852054044218e-07,
            0.5251518455536499,
            -0.668710345667336,
            0.6067086443974802,
            -0.606711497194891,
            0.6687148024583701,
            -0.5251586523105128,
        ]
        dist = np.linalg.norm(np.array(current_positions) - np.array(init_positions))
        if dist > 0.2:
            self.goto_target(INIT_HEAD_POSE, antennas=[0.0, 0.0], duration=1)
            time.sleep(0.2)

        # Pfiou
        self.play_sound("go_sleep.wav")

        # Move to the sleep position
        self._goto_joint_positions(
            head_joint_positions=SLEEP_HEAD_JOINT_POSITIONS,
            antennas_joint_positions=SLEEP_ANTENNAS_JOINT_POSITIONS,
            duration=2,
        )
        self._last_head_pose = SLEEP_HEAD_POSE
        time.sleep(2)

    def look_at_image(self, u: int, v: int, duration: float = 1.0) -> None:
        """Make the robot head look at a point defined by a pixel position (u,v).

        # TODO image of reachy mini coordinate system

        Args:
            u (int): Horizontal coordinate in image frame.
            v (int): Vertical coordinate in image frame.
            duration (float): Duration of the movement in seconds. If 0, the head will snap to the position immediately.

        Raises:
            ValueError: If duration is negative.

        """
        assert 0 < u < IMAGE_SIZE[0], f"u must be in [0, {IMAGE_SIZE[0]}], got {u}."
        assert 0 < v < IMAGE_SIZE[1], f"v must be in [0, {IMAGE_SIZE[1]}], got {v}."

        if duration < 0:
            raise ValueError("Duration can't be negative.")

        x_n, y_n = cv2.undistortPoints(np.float32([[[u, v]]]), self.K, self.D)[0, 0]  # type: ignore

        ray_cam = np.array([x_n, y_n, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)

        T_world_head = self._get_current_head_pose()
        T_world_cam = T_world_head @ self.T_head_cam

        R_wc = T_world_cam[:3, :3]
        t_wc = T_world_cam[:3, 3]

        ray_world = R_wc @ ray_cam

        P_world = t_wc + ray_world

        self.look_at_world(*P_world, duration=duration)

    def look_at_world(self, x: float, y: float, z: float, duration: float = 1.0):
        """Look at a specific point in 3D space in Reachy Mini's reference frame.

        TODO include image of reachy mini coordinate system

        Args:
            x (float): X coordinate in meters.
            y (float): Y coordinate in meters.
            z (float): Z coordinate in meters.
            duration (float): Duration of the movement in seconds. If 0, the head will snap to the position immediately.

        Raises:
            ValueError: If duration is negative.

        """
        if duration < 0:
            raise ValueError("Duration can't be negative.")

        # Head is at the origin, so vector from head to target position is directly the target position
        # TODO FIX : Actually, the head frame is not the origin frame wrt the kinematics. Close enough for now.
        target_position = np.array([x, y, z])
        target_vector = target_position / np.linalg.norm(
            target_position
        )  # normalize the vector

        # head_pointing straight vector
        straight_head_vector = np.array([1, 0, 0])

        # Calculate the rotation needed to align the head with the target vector
        v1 = straight_head_vector
        v2 = target_vector
        axis = np.cross(v1, v2)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            # Vectors are (almost) parallel
            if np.dot(v1, v2) > 0:
                rot_mat = np.eye(3)
            else:
                # Opposite direction: rotate 180° around any perpendicular axis
                perp = np.array([0, 1, 0]) if abs(v1[0]) < 0.9 else np.array([0, 0, 1])
                axis = np.cross(v1, perp)
                axis /= np.linalg.norm(axis)
                rot_mat = R.from_rotvec(np.pi * axis).as_matrix()
        else:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
            rotation_vector = angle * axis
            rot_mat = R.from_rotvec(rotation_vector).as_matrix()

        target_head_pose = np.eye(4)
        target_head_pose[:3, :3] = rot_mat

        # If duration is specified, use the goto_target method to move smoothly
        # Otherwise, set the position immediately
        if duration > 0:
            self.goto_target(target_head_pose, duration=duration)
        else:
            self.set_target(target_head_pose)

    # Multimedia methods
    def play_sound(self, sound_file: str) -> None:
        """Play a sound file from the assets directory.

        Args:
            sound_file (str): The name of the sound file to play (e.g., "proud2.wav").

        """
        if pygame.mixer is None:
            print("Pygame mixer is not initialized. Cannot play sound.")
            return
        pygame.mixer.music.load(f"{ReachyMini.assets_root_path}/{sound_file}")
        pygame.mixer.music.play()

    def _goto_joint_positions(
        self,
        head_joint_positions: Optional[
            List[float]
        ] = None,  # [yaw, stewart_platform x 6] length 7
        antennas_joint_positions: Optional[
            List[float]
        ] = None,  # [left_angle, right_angle] length 2
        duration: float = 0.5,  # Duration in seconds for the movement
    ) -> None:
        """Go to a target head joint positions and/or antennas joint positions using joint space interpolation, in "duration" seconds.

        [Internal] Go to a target head joint positions and/or antennas joint positions using joint space interpolation, in "duration" seconds.

        Args:
            head_joint_positions (Optional[List[float]]): List of head joint positions in radians (length 7).
            antennas_joint_positions (Optional[List[float]]): List of antennas joint positions in radians (length 2).
            duration (float): Duration of the movement in seconds. Default is 0.5 seconds.

        Raises:
            ValueError: If neither head_joint_positions nor antennas_joint_positions are provided, or if duration is not positive.

        """
        if duration <= 0.0:
            raise ValueError(
                "Duration must be positive and non-zero. Use set_target() for immediate position setting."
            )

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

            self._set_joint_positions(list(head_joint), list(antennas_joint))
            time.sleep(0.01)

    def _get_current_joint_positions(self) -> tuple[list[float], list[float]]:
        """Get the current joint positions of the head and antennas.

        [Internal] Get the current joint positions of the head and antennas (in rad)

        Returns:
            tuple: A tuple containing two lists:
                - List of head joint positions (rad) (length 7).
                - List of antennas joint positions (rad) (length 2).

        """
        return self.client.get_current_joints()

    def _get_current_head_pose(self) -> np.ndarray:
        """Get the current head pose as a 4x4 matrix.

        [Internal] Get the current head pose as a 4x4 matrix.

        Returns:
            np.ndarray: A 4x4 matrix representing the current head pose.

        """
        return self.client.get_current_head_pose()

    def _set_joint_positions(
        self,
        head_joint_positions: Optional[
            List[float]
        ],  # [yaw, stewart_platform x 6] length 7
        antennas_joint_positions: Optional[
            List[float]
        ],  # [left_angle, right_angle] length 2
    ):
        """Set the joint positions of the head and/or antennas.

        [Internal] Set the joint positions of the head and/or antennas.

        Args:
            head_joint_positions (Optional[List[float]]): List of head joint positions in radians (length 7).
            antennas_joint_positions (Optional[List[float]]): List of antennas joint positions in radians (length 2).

        """
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

    def _set_head_pose(self, pose: np.ndarray, body_yaw: float = 0.0) -> None:
        """Set the head pose to a specific 4x4 matrix.

        Args:
            pose (np.ndarray): A 4x4 matrix representing the desired head pose.
            body_yaw (float): The yaw angle of the body, used to adjust the head pose.

        Raises:
            ValueError: If the shape of the pose is not (4, 4).

        """
        cmd = {}

        if pose is not None:
            assert pose.shape == (4, 4), (
                f"Head pose should be a 4x4 matrix, got {pose.shape}."
            )
            cmd["head_pose"] = pose.tolist()
        else:
            raise ValueError("Pose must be provided as a 4x4 matrix.")

        cmd["body_yaw"] = body_yaw

        self.client.send_command(json.dumps(cmd))

    def _set_head_operation_mode(self, mode: int) -> None:
        """Set the operation mode for the head motors.

        Args:
            mode (int): The desired operation mode.

        """
        self.client.send_command(json.dumps({"head_operation_mode": mode}))

    def _set_antennas_operation_mode(self, mode: int) -> None:
        """Set the operation mode for the antennas motors.

        Args:
            mode (int): The desired operation mode.

        """
        self.client.send_command(json.dumps({"antennas_operation_mode": mode}))

    def enable_motors(self) -> None:
        """Enable the motors."""
        self._set_torque(True)

    def disable_motors(self) -> None:
        """Disable the motors."""
        self._set_torque(False)

    def make_motors_compliant(
        self,
        head: Optional[bool] = None,
        antennas: Optional[bool] = None,
        compensate_gravity: bool = False,
    ) -> None:
        """Set the head and/or antennas to compliant mode. This means that the motors will not resist external forces and will allow free movement.

            The compensate_gravity argument will enable gravity compensation for the head motors if they are in the compliant mode,
            In the non-compliant mode, the gravity compensation will have no effect.

        Args:
            head (bool): If True, set the head to compliant mode.
            antennas (bool): If True, set the antennas to compliant mode.
            compensate_gravity (bool): If True, enable gravity compensation for the head motors.

        """
        if head is not None:
            self._set_head_operation_mode(
                0 if head else 3
            )  # 0 is compliant mode, 3 is position control mode

        if antennas is not None:
            self._set_antennas_operation_mode(
                0 if antennas else 3
            )  # 0 is compliant mode, 3 is position control mode

        if compensate_gravity:
            self.enable_gravity_compensation()
        else:
            self.disable_gravity_compensation()

    def _set_torque(self, on: bool):
        self.client.send_command(json.dumps({"torque": on}))

    def _set_head_joint_current(self, current: List[int]) -> None:
        """Set the head joint current (torque) in milliamperes (mA).

        Args:
            current (List[int]): A list of joint currents for the head.

        """
        assert len(current) == 7, (
            f"Head joint current must have length 7, got {current}."
        )
        self.client.send_command(json.dumps({"head_joint_current": list(current)}))

    def _set_check_collision(self, check: bool) -> None:
        """Set whether to check for collisions.

        Args:
            check (bool): If True, the backend will check for collisions.

        """
        self.client.send_command(json.dumps({"check_collision": check}))

    def enable_gravity_compensation(self) -> None:
        """Enable gravity compensation for the head motors."""
        self.client.send_command(json.dumps({"gravity_compensation": True}))

    def disable_gravity_compensation(self) -> None:
        """Disable gravity compensation for the head motors."""
        self.client.send_command(json.dumps({"gravity_compensation": False}))

    def set_automatic_body_yaw(self, body_yaw: float) -> None:
        """Set the automatic body yaw.

        Args:
            body_yaw (float): The yaw angle of the body in radians.

        """
        self.client.send_command(json.dumps({"automatic_body_yaw": body_yaw}))
