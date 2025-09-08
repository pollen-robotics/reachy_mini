"""Base class for robot backends, simulated or real.

This module defines the `Backend` class, which serves as a base for implementing
different types of robot backends, whether they are simulated (like Mujoco) or real
(connected via serial port). The class provides methods for managing joint positions,
torque control, and other backend-specific functionalities.
It is designed to be extended by subclasses that implement the specific behavior for
each type of backend.
"""

import asyncio
import json
import logging
import os
import threading
import time
from abc import abstractmethod
from enum import Enum
from importlib.resources import files
from typing import List

from reachy_mini.motion.goto import GotoMove

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import numpy as np
import pygame
from scipy.spatial.transform import Rotation as R

import reachy_mini
from reachy_mini.kinematics import NNKinematics, PlacoKinematics
from reachy_mini.motion.move import Move
from reachy_mini.utils.interpolation import (
    InterpolationTechnique,
    distance_between_poses,
    time_trajectory,
)

try:
    pygame.mixer.init()
except pygame.error as e:
    print(f"Failed to initialize pygame mixer: {e}")
    pygame.mixer = None


class MotorControlMode(str, Enum):
    """Enum for motor control modes."""

    Enabled = "enabled"  # Torque ON and controlled in position
    Disabled = "disabled"  # Torque OFF
    GravityCompensation = "gravity_compensation"  # Torque ON and controlled in current to compensate for gravity


class Backend:
    """Base class for robot backends, simulated or real."""

    urdf_root_path: str = str(
        files(reachy_mini).joinpath("descriptions/reachy_mini/urdf")
    )

    assets_root_path: str = str(files(reachy_mini).joinpath("assets"))
    models_root_path: str = str(files(reachy_mini).joinpath("assets/models"))

    def __init__(
        self,
        log_level: str = "INFO",
        check_collision: bool = False,
        kinematics_engine: str = "Placo",
    ) -> None:
        """Initialize the backend."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        self.should_stop = threading.Event()
        self.ready = threading.Event()

        self.check_collision = (
            check_collision  # Flag to enable/disable collision checking
        )
        self.kinematics_engine = kinematics_engine
        assert self.kinematics_engine != "Analytical", (
            "Analytical kinematics engine is not integrated yet"
        )

        self.logger.info(f"Using {self.kinematics_engine} kinematics engine")

        if self.check_collision:
            assert self.kinematics_engine == "Placo", (
                "Collision checking is only available with Placo Kinematics"
            )

        self.gravity_compensation_mode = False  # Flag for gravity compensation mode

        if self.gravity_compensation_mode:
            assert self.kinematics_engine == "Placo", (
                "Gravity compensation is only available with Placo kinematics"
            )

        if self.kinematics_engine == "Placo":
            self.head_kinematics = PlacoKinematics(
                Backend.urdf_root_path, check_collision=self.check_collision
            )
        elif self.kinematics_engine == "NN":
            self.head_kinematics = NNKinematics(Backend.models_root_path)
        else:
            print("???")
            exit()

        self.current_head_pose = None  # 4x4 pose matrix
        self.target_head_pose = None  # 4x4 pose matrix
        self.target_body_yaw = None  # Last body yaw used in IK computations

        self.target_head_joint_positions = None  # [yaw, 0, 1, 2, 3, 4, 5]
        self.current_head_joint_positions = None  # [yaw, 0, 1, 2, 3, 4, 5]
        self.target_antenna_joint_positions = None  # [0, 1]
        self.current_antenna_joint_positions = None  # [0, 1]

        self.joint_positions_publisher = None  # Placeholder for a publisher object
        self.pose_publisher = None  # Placeholder for a pose publisher object
        self.recording_publisher = None  # Placeholder for a recording publisher object
        self.error = None  # To store any error that occurs during execution
        self.is_recording = False  # Flag to indicate if recording is active
        self.recorded_data = []  # List to store recorded data

        # variables to store the last computed head joint positions and pose
        self._last_target_body_yaw = None  # Last body yaw used in IK computations
        self._last_target_head_pose = None  # Last head pose used in IK computations
        self.target_head_joint_current = None  # Placeholder for head joint torque
        self.target_head_operation_mode = None  # Placeholder for head operation mode
        self.ik_required = False  # Flag to indicate if IK computation is required

        # Tolerance for kinematics computations
        # For Forward kinematics (around 0.25deg)
        # - FK is calculated at each timestep and is susceptible to noise
        self._fk_kin_tolerance = 1e-3  # rads
        # For Inverse kinematics (around 0.5mm and 0.1 degrees)
        # - IK is calculated only when the head pose is set by the user
        self._ik_kin_tolerance = {
            "rad": 2e-3,  # rads
            "m": 0.5e-3,  # m
        }

        # Recording lock to guard buffer swaps and appends
        self._rec_lock = threading.Lock()

    # Life cycle methods
    def wrapped_run(self):
        """Run the backend in a try-except block to store errors."""
        try:
            self.run()
        except Exception as e:
            self.error = str(e)
            self.close()
            raise e

    def run(self):
        """Run the backend.

        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError("The method run should be overridden by subclasses.")

    def close(self) -> None:
        """Close the backend.

        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The method close should be overridden by subclasses."
        )

    def get_status(self):
        """Return backend statistics.

        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The method get_status should be overridden by subclasses."
        )

    # Present/Target joint positions
    def set_joint_positions_publisher(self, publisher) -> None:
        """Set the publisher for joint positions.

        Args:
            publisher: A publisher object that will be used to publish joint positions.

        """
        self.joint_positions_publisher = publisher

    def set_pose_publisher(self, publisher) -> None:
        """Set the publisher for head pose.

        Args:
            publisher: A publisher object that will be used to publish head pose.

        """
        self.pose_publisher = publisher

    def update_target_head_joints_from_ik(
        self, pose: np.ndarray | None = None, body_yaw: float | None = None
    ) -> None:
        """Update the target head joint positions from inverse kinematics.

        Args:
            pose (np.ndarray): 4x4 pose matrix representing the head pose.
            body_yaw (float): The yaw angle of the body, used to adjust the head pose.

        """
        if pose is None:
            pose = (
                self.target_head_pose
                if self.target_head_pose is not None
                else np.eye(4)
            )

        if body_yaw is None:
            body_yaw = self.target_body_yaw if self.target_body_yaw is not None else 0.0

        # Compute the inverse kinematics to get the head joint positions
        joints = self.head_kinematics.ik(pose, body_yaw=body_yaw)

        if joints is None or np.any(np.isnan(joints)):
            raise ValueError("WARNING: Collision detected or head pose not achievable!")

        # update the target head pose and body yaw
        self._last_target_head_pose = pose
        self._last_target_body_yaw = body_yaw

        self.target_head_joint_positions = joints

    def set_target_head_pose(
        self,
        pose: np.ndarray,
        body_yaw: float = 0.0,
    ) -> None:
        """Set the target head pose for the robot.

        Args:
            pose (np.ndarray): 4x4 pose matrix representing the head pose.
            body_yaw (float): The yaw angle of the body, used to adjust the head pose.

        """
        self.target_head_pose = pose
        self.target_body_yaw = body_yaw
        self.ik_required = True

    def set_target_head_joint_positions(self, positions: List[float]) -> None:
        """Set the head joint positions.

        Args:
            positions (List[float]): A list of joint positions for the head.

        """
        self.target_head_joint_positions = positions
        self.ik_required = False

    def set_target(
        self,
        head: np.ndarray | None = None,  # 4x4 pose matrix
        antennas: np.ndarray
        | list[float]
        | None = None,  # [left_angle, right_angle] (in rads)
        body_yaw: float = 0.0,  # Body yaw angle in radians
    ) -> None:
        """Set the target head pose and/or antenna positions."""
        if head is not None:
            self.set_target_head_pose(head, body_yaw)

        if antennas is not None:
            if isinstance(antennas, np.ndarray):
                antennas = antennas.tolist()
            self.set_target_antenna_joint_positions(antennas)

    def set_target_antenna_joint_positions(
        self,
        positions: List[float],
    ) -> None:
        """Set the antenna joint positions.

        Args:
            positions (List[float]): A list of joint positions for the antenna.

        """
        self.target_antenna_joint_positions = positions

    def set_target_head_joint_current(self, current: List[float]) -> None:
        """Set the head joint current.

        Args:
            current (List[float]): A list of current values for the head motors.

        """
        self.target_head_joint_current = current
        self.ik_required = False

    async def play_move(self, move: Move, play_frequency: float = 100.0) -> None:
        """Asynchronously play a Move.

        Args:
            move (Move): The Move object to be played.
            play_frequency (float): The frequency at which to evaluate the move (in Hz).

        """
        sleep_period = 1.0 / play_frequency

        t0 = time.time()
        while time.time() - t0 < move.duration:
            t = time.time() - t0

            head, antennas, body_yaw = move.evaluate(t)
            if head is not None:
                self.set_target_head_pose(
                    head,
                    body_yaw=body_yaw if body_yaw is not None else 0.0,
                )
            if antennas is not None:
                self.set_target_antenna_joint_positions(list(antennas))

            elapsed = time.time() - t0 - t
            if elapsed < sleep_period:
                await asyncio.sleep(sleep_period - elapsed)
            else:
                await asyncio.sleep(0.001)

    async def goto_target(
        self,
        head: np.ndarray | None = None,  # 4x4 pose matrix
        antennas: np.ndarray
        | list[float]
        | None = None,  # [left_angle, right_angle] (in rads)
        duration: float = 0.5,  # Duration in seconds for the movement, default is 0.5 seconds.
        method: InterpolationTechnique = InterpolationTechnique.MIN_JERK,  # can be "linear", "minjerk", "ease" or "cartoon", default is "minjerk"
        body_yaw: float = 0.0,  # Body yaw angle in radians
    ):
        """Asynchronously go to a target head pose and/or antennas position using task space interpolation, in "duration" seconds.

        Args:
            head (np.ndarray | None): 4x4 pose matrix representing the target head pose.
            antennas (np.ndarray | list[float] | None): 1D array with two elements representing the angles of the antennas in radians.
            duration (float): Duration of the movement in seconds.
            method (str): Interpolation method to use ("linear", "minjerk", "ease", "cartoon"). Default is "minjerk".
            body_yaw (float): Body yaw angle in radians.

        Raises:
            ValueError: If neither head nor antennas are provided, or if duration is not positive.

        """
        return await self.play_move(
            move=GotoMove(
                start_head_pose=self.get_present_head_pose(),
                target_head_pose=head,
                start_body_yaw=self.get_present_body_yaw(),
                target_body_yaw=body_yaw,
                start_antennas=np.array(self.get_present_antenna_joint_positions()),
                target_antennas=np.array(antennas) if antennas is not None else None,
                duration=duration,
                method=method,
            )
        )

    async def goto_joint_positions(
        self,
        head_joint_positions: list[float]
        | None = None,  # [yaw, stewart_platform x 6] length 7
        antennas_joint_positions: list[float]
        | None = None,  # [left_angle, right_angle] length 2
        duration: float = 0.5,  # Duration in seconds for the movement
        method: InterpolationTechnique = InterpolationTechnique.MIN_JERK,  # can be "linear", "minjerk", "ease" or "cartoon", default is "minjerk"
    ) -> None:
        """Asynchronously go to a target head joint positions and/or antennas joint positions using joint space interpolation, in "duration" seconds.

        Go to a target head joint positions and/or antennas joint positions using joint space interpolation, in "duration" seconds.

        Args:
            head_joint_positions (Optional[List[float]]): List of head joint positions in radians (length 7).
            antennas_joint_positions (Optional[List[float]]): List of antennas joint positions in radians (length 2).
            duration (float): Duration of the movement in seconds. Default is 0.5 seconds.
            method (str): Interpolation method to use ("linear", "minjerk", "ease", "cartoon"). Default is "minjerk".

        Raises:
            ValueError: If neither head_joint_positions nor antennas_joint_positions are provided, or if duration is not positive.

        """
        if duration <= 0.0:
            raise ValueError(
                "Duration must be positive and non-zero. Use set_target() for immediate position setting."
            )

        start_head = np.array(self.get_present_head_joint_positions())
        start_antennas = np.array(self.get_present_antenna_joint_positions())

        target_head = (
            np.array(head_joint_positions)
            if head_joint_positions is not None
            else start_head
        )
        target_antennas = (
            np.array(antennas_joint_positions)
            if antennas_joint_positions is not None
            else start_antennas
        )

        t0 = time.time()
        while time.time() - t0 < duration:
            t = time.time() - t0

            interp_time = time_trajectory(t / duration, method=method)

            head_joint = start_head + (target_head - start_head) * interp_time
            antennas_joint = (
                start_antennas + (target_antennas - start_antennas) * interp_time
            )

            self.set_target_head_joint_positions(head_joint.tolist())
            self.set_target_antenna_joint_positions(antennas_joint.tolist())
            await asyncio.sleep(0.01)

    def set_recording_publisher(self, publisher) -> None:
        """Set the publisher for recording data.

        Args:
            publisher: A publisher object that will be used to publish recorded data.

        """
        self.recording_publisher = publisher

    def append_record(self, record: dict) -> None:
        """Append a record to the recorded data.

        Args:
            record (dict): A dictionary containing the record data to be appended.

        """
        if not self.is_recording:
            return
        # Double-check under lock to avoid race with stop_recording
        with self._rec_lock:
            if self.is_recording:
                self.recorded_data.append(record)

    def start_recording(self) -> None:
        """Start recording data."""
        with self._rec_lock:
            self.recorded_data = []
            self.is_recording = True

    def stop_recording(self) -> None:
        """Stop recording data and publish the recorded data."""
        # Swap buffer under lock so writers cannot touch the published list
        with self._rec_lock:
            self.is_recording = False
            recorded_data, self.recorded_data = self.recorded_data, []
        # Publish outside the lock
        if self.recording_publisher is not None:
            self.recording_publisher.put(json.dumps(recorded_data))
        else:
            self.logger.warning(
                "stop_recording called but recording_publisher is not set; dropping data."
            )

    def get_present_head_joint_positions(self) -> List[float]:
        """Return the present head joint positions.

        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The method get_present_head_joint_positions should be overridden by subclasses."
        )

    def get_present_body_yaw(self) -> float:
        """Return the present body yaw."""
        return self.get_present_head_joint_positions()[0]

    def get_present_head_pose(self) -> np.ndarray:
        """Return the present head pose as a 4x4 matrix."""
        assert self.current_head_pose is not None, (
            "The current head pose is not set. Please call the update_head_kinematics_model method first."
        )
        return self.current_head_pose

    def get_current_head_pose(self) -> np.ndarray:
        """Return the present head pose as a 4x4 matrix."""
        return self.get_present_head_pose()

    def get_present_antenna_joint_positions(self) -> List[float]:
        """Return the present antenna joint positions.

        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The method get_present_antenna_joint_positions should be overridden by subclasses."
        )

    # Kinematics methods
    def update_head_kinematics_model(
        self,
        head_joint_positions: List[float] | None = None,
        antennas_joint_positions: List[float] | None = None,
    ) -> None:
        """Update the placo kinematics of the robot.

        Args:
            head_joint_positions (List[float] | None): The joint positions of the head.
            antennas_joint_positions (List[float] | None): The joint positions of the antennas.

        Returns:
            None: This method does not return anything.

        This method updates the head kinematics model with the given joint positions.
        - If the joint positions are not provided, it will use the current joint positions.
        - If the head joint positions have not changed, it will return without recomputing the forward kinematics.
        - If the head joint positions have changed, it will compute the forward kinematics to get the current head pose.
        - If the forward kinematics fails, it will raise an assertion error.
        - If the antennas joint positions are provided, it will update the current antenna joint positions.

        Note:
            This method will update the `current_head_pose` and `current_head_joint_positions`
            attributes of the backend instance with the computed values. And the `current_antenna_joint_positions` if provided.

        """
        if head_joint_positions is None:
            head_joint_positions = self.get_present_head_joint_positions()

        # Compute the forward kinematics to get the current head pose
        self.current_head_pose = self.head_kinematics.fk(head_joint_positions)

        # Check if the FK was successful
        assert self.current_head_pose is not None, (
            "FK failed to compute the current head pose."
        )

        # Store the last head joint positions
        self.current_head_joint_positions = head_joint_positions

        if antennas_joint_positions is not None:
            self.current_antenna_joint_positions = antennas_joint_positions

    def set_automatic_body_yaw(self, body_yaw: float) -> None:
        """Set the automatic body yaw.

        Args:
            body_yaw (float): The yaw angle of the body.

        """
        self.head_kinematics.start_body_yaw = body_yaw

    # Multimedia methods
    def play_sound(self, sound_file: str) -> None:
        """Play a sound file from the assets directory.

        If the file is not found in the assets directory, try to load the path itself.

        Args:
            sound_file (str): The name of the sound file to play (e.g., "proud2.wav").

        """
        if pygame.mixer is None:
            print("Pygame mixer is not initialized. Cannot play sound.")
            return

        # first check if the name exists in the asset sound directory
        file_path = f"{self.assets_root_path}/{sound_file}"
        if not os.path.exists(file_path):
            # If not, check if the raw_path exists
            if not os.path.exists(sound_file):
                raise FileNotFoundError(f"Sound file {sound_file} not found.")
            else:
                file_path = sound_file

        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

    # Basic move definitions
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

    async def wake_up(self) -> None:
        """Wake up the robot - go to the initial head position and play the wake up emote and sound."""
        await asyncio.sleep(0.1)

        _, _, magic_distance = distance_between_poses(
            self.get_current_head_pose(), self.INIT_HEAD_POSE
        )

        await self.goto_target(
            self.INIT_HEAD_POSE,
            antennas=[0.0, 0.0],
            duration=magic_distance * 20 / 1000,  # ms_per_magic_mm = 10
        )
        await asyncio.sleep(0.1)

        # Toudoum
        self.play_sound("proud2.wav")

        # Roll 20° to the left
        pose = self.INIT_HEAD_POSE.copy()
        pose[:3, :3] = R.from_euler("xyz", [20, 0, 0], degrees=True).as_matrix()
        await self.goto_target(pose, duration=0.2)

        # Go back to the initial position
        await self.goto_target(self.INIT_HEAD_POSE, duration=0.2)

    async def goto_sleep(self) -> None:
        """Put the robot to sleep by moving the head and antennas to a predefined sleep position.

        - If we are already very close to the sleep position, we do nothing.
        - If we are far from the sleep position:
            - If we are far from the initial position, we move there first.
            - If we are close to the initial position, we move directly to the sleep position.
        """
        # Magic units
        _, _, dist_to_sleep_pose = distance_between_poses(
            self.get_current_head_pose(), self.SLEEP_HEAD_POSE
        )
        _, _, dist_to_init_pose = distance_between_poses(
            self.get_current_head_pose(), self.INIT_HEAD_POSE
        )

        sleep_time = 2.0

        # Thresholds found empirically.
        if dist_to_sleep_pose > 10:
            if dist_to_init_pose > 30:
                # Move to the initial position
                await self.goto_target(
                    self.INIT_HEAD_POSE, antennas=[0.0, 0.0], duration=1
                )
                await asyncio.sleep(0.2)

            self.play_sound("go_sleep.wav")

            # Move to the sleep position
            await self.goto_target(
                self.SLEEP_HEAD_POSE,
                antennas=self.SLEEP_ANTENNAS_JOINT_POSITIONS,
                duration=2,
            )
        else:
            # The sound doesn't play fully if we don't wait enough
            self.play_sound("go_sleep.wav")
            sleep_time += 3

        self._last_head_pose = self.SLEEP_HEAD_POSE
        await asyncio.sleep(sleep_time)

    # Motor control modes
    @abstractmethod
    def get_motor_control_mode(self) -> MotorControlMode:
        """Get the motor control mode."""
        pass

    @abstractmethod
    def set_motor_control_mode(self, mode: MotorControlMode) -> None:
        """Set the motor control mode."""
        pass
