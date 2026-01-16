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
import threading
import time
import typing
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Dict, Optional

import numpy as np
import zenoh
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

if typing.TYPE_CHECKING:
    from reachy_mini.daemon.backend.mockup_sim.backend import MockupSimBackendStatus
    from reachy_mini.daemon.backend.mujoco.backend import MujocoBackendStatus
    from reachy_mini.daemon.backend.robot.backend import RobotBackendStatus
    from reachy_mini.kinematics import AnyKinematics
from reachy_mini.media.media_manager import MediaBackend, MediaManager
from reachy_mini.motion.goto import GotoMove
from reachy_mini.motion.move import Move
from reachy_mini.utils.constants import MODELS_ROOT_PATH, URDF_ROOT_PATH
from reachy_mini.utils.interpolation import (
    InterpolationTechnique,
    distance_between_poses,
    time_trajectory,
)


class MotorControlMode(str, Enum):
    """Enum for motor control modes."""

    Enabled = "enabled"  # Torque ON and controlled in position
    Disabled = "disabled"  # Torque OFF
    GravityCompensation = "gravity_compensation"  # Torque ON and controlled in current to compensate for gravity


class Backend:
    """Base class for robot backends, simulated or real."""

    def __init__(
        self,
        log_level: str = "INFO",
        check_collision: bool = False,
        kinematics_engine: str = "AnalyticalKinematics",
        use_audio: bool = True,
        wireless_version: bool = False,
    ) -> None:
        """Initialize the backend."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        self.use_audio = use_audio

        self.should_stop = threading.Event()
        self.ready = threading.Event()

        self.check_collision = (
            check_collision  # Flag to enable/disable collision checking
        )
        self.kinematics_engine = kinematics_engine

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
            from reachy_mini.kinematics import PlacoKinematics

            self.head_kinematics: AnyKinematics = PlacoKinematics(
                URDF_ROOT_PATH, check_collision=self.check_collision
            )
        elif self.kinematics_engine == "NN":
            from reachy_mini.kinematics import NNKinematics

            self.head_kinematics = NNKinematics(MODELS_ROOT_PATH)
        elif self.kinematics_engine == "AnalyticalKinematics":
            from reachy_mini.kinematics import AnalyticalKinematics

            self.head_kinematics = AnalyticalKinematics()
        else:
            raise ValueError(
                f"Unknown kinematics engine: {self.kinematics_engine}. Use 'Placo', 'NN' or 'AnalyticalKinematics'."
            )

        self.current_head_pose: Annotated[NDArray[np.float64], (4, 4)] | None = (
            None  # 4x4 pose matrix
        )
        self.target_head_pose: Annotated[NDArray[np.float64], (4, 4)] | None = (
            None  # 4x4 pose matrix
        )
        self.target_body_yaw: float | None = (
            None  # Last body yaw used in IK computations
        )

        self.target_head_joint_positions: (
            Annotated[NDArray[np.float64], (7,)] | None
        ) = None  # [yaw, 0, 1, 2, 3, 4, 5]
        self.current_head_joint_positions: (
            Annotated[NDArray[np.float64], (7,)] | None
        ) = None  # [yaw, 0, 1, 2, 3, 4, 5]
        self.target_antenna_joint_positions: (
            Annotated[NDArray[np.float64], (2,)] | None
        ) = None  # [0, 1]
        self.current_antenna_joint_positions: (
            Annotated[NDArray[np.float64], (2,)] | None
        ) = None  # [0, 1]

        self.joint_positions_publisher: zenoh.Publisher | None = None
        self.pose_publisher: zenoh.Publisher | None = None
        self.recording_publisher: zenoh.Publisher | None = None
        self.imu_publisher: zenoh.Publisher | None = None
        self.error: str | None = None  # To store any error that occurs during execution
        self.is_recording = False  # Flag to indicate if recording is active
        self.recorded_data: list[dict[str, Any]] = []  # List to store recorded data

        # variables to store the last computed head joint positions and pose
        self._last_target_body_yaw: float | None = (
            None  # Last body yaw used in IK computations
        )
        self._last_target_head_pose: Annotated[NDArray[np.float64], (4, 4)] | None = (
            None  # Last head pose used in IK computations
        )
        self.target_head_joint_current: Annotated[NDArray[np.float64], (7,)] | None = (
            None  # Placeholder for head joint torque
        )
        self.ik_required = False  # Flag to indicate if IK computation is required

        self.is_shutting_down = False

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

        self.audio: Optional[MediaManager] = None
        if self.use_audio:
            if wireless_version:
                self.logger.debug(
                    "Initializing daemon audio backend for wireless version."
                )
                self.audio = MediaManager(
                    backend=MediaBackend.GSTREAMER_NO_VIDEO, log_level=log_level
                )
            else:
                self.logger.debug(
                    "Initializing daemon audio backend for non-wireless version."
                )
                self.audio = MediaManager(
                    backend=MediaBackend.DEFAULT_NO_VIDEO, log_level=log_level
                )

        # Guard to ensure only one play_move/goto is executed at a time (goto itself uses play_move, so we need an RLock)
        self._play_move_lock = threading.RLock()
        self._active_move_depth = (
            0  # Tracks nested acquisitions within the owning thread
        )

    # Life cycle methods
    def wrapped_run(self) -> None:
        """Run the backend in a try-except block to store errors."""
        try:
            self.run()
        except Exception as e:
            self.error = str(e)
            self.close()
            raise e

    def run(self) -> None:
        """Run the backend.

        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError("The method run should be overridden by subclasses.")

    def close(self) -> None:
        """Close the backend and release resources.

        Subclasses should override this method to add their own cleanup logic,
        and call super().close() at the end to ensure audio resources are released.

        Note: This base implementation handles common cleanup (audio).
        Subclasses must still implement their own cleanup for backend-specific resources.
        """
        self.logger.debug("Backend.close() - cleaning up audio resources")
        if self.audio is not None:
            self.audio.close()
            self.audio = None

    @property
    def is_move_running(self) -> bool:
        """Return True if a move is currently executing."""
        return self._active_move_depth > 0

    def _try_start_move(self) -> bool:
        """Attempt to acquire the move guard, returning False if another client already owns it."""
        if not self._play_move_lock.acquire(blocking=False):
            return False
        self._active_move_depth += 1
        return True

    def _end_move(self) -> None:
        """Release the move guard; paired with every successful _try_start_move()."""
        if self._active_move_depth > 0:
            self._active_move_depth -= 1
        self._play_move_lock.release()

    def get_status(
        self,
    ) -> "RobotBackendStatus | MujocoBackendStatus | MockupSimBackendStatus":
        """Return backend statistics.

        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The method get_status should be overridden by subclasses."
        )

    # Present/Target joint positions
    def set_joint_positions_publisher(self, publisher: zenoh.Publisher) -> None:
        """Set the publisher for joint positions.

        Args:
            publisher: A publisher object that will be used to publish joint positions.

        """
        self.joint_positions_publisher = publisher

    def set_pose_publisher(self, publisher: zenoh.Publisher) -> None:
        """Set the publisher for head pose.

        Args:
            publisher: A publisher object that will be used to publish head pose.

        """
        self.pose_publisher = publisher

    def set_imu_publisher(self, publisher: zenoh.Publisher) -> None:
        """Set the publisher for IMU data.

        Args:
            publisher: A publisher object that will be used to publish IMU data.

        """
        self.imu_publisher = publisher

    def update_target_head_joints_from_ik(
        self,
        pose: Annotated[NDArray[np.float64], (4, 4)] | None = None,
        body_yaw: float | None = None,
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
        pose: Annotated[NDArray[np.float64], (4, 4)],
    ) -> None:
        """Set the target head pose for the robot.

        Args:
            pose (np.ndarray): 4x4 pose matrix representing the head pose.

        """
        self.target_head_pose = pose
        self.ik_required = True

    def set_target_body_yaw(self, body_yaw: float) -> None:
        """Set the target body yaw for the robot.

        Only used when doing a set_target() with a standalone body_yaw (no head pose).

        Args:
            body_yaw (float): The yaw angle of the body

        """
        self.target_body_yaw = body_yaw
        self.ik_required = True  # Do we need that here?

    def set_target_head_joint_positions(
        self, positions: Annotated[NDArray[np.float64], (7,)] | None
    ) -> None:
        """Set the head joint positions.

        Args:
            positions (List[float]): A list of joint positions for the head.

        """
        self.target_head_joint_positions = positions
        self.ik_required = False

    def set_target(
        self,
        head: Annotated[NDArray[np.float64], (4, 4)] | None = None,  # 4x4 pose matrix
        antennas: Annotated[NDArray[np.float64], (2,)]
        | None = None,  # [right_angle, left_angle] (in rads)
        body_yaw: float | None = None,  # Body yaw angle in radians
    ) -> None:
        """Set the target head pose and/or antenna positions and/or body_yaw."""
        if head is not None:
            self.set_target_head_pose(head)

        if body_yaw is not None:
            self.set_target_body_yaw(body_yaw)

        if antennas is not None:
            self.set_target_antenna_joint_positions(antennas)

    def set_target_antenna_joint_positions(
        self,
        positions: Annotated[NDArray[np.float64], (2,)],
    ) -> None:
        """Set the antenna joint positions.

        Args:
            positions (List[float]): A list of joint positions for the antenna.

        """
        self.target_antenna_joint_positions = positions

    def set_target_head_joint_current(
        self,
        current: Annotated[NDArray[np.float64], (7,)],
    ) -> None:
        """Set the head joint current.

        Args:
            current (Annotated[NDArray[np.float64], (7,)]): A list of current values for the head motors.

        """
        self.target_head_joint_current = current
        self.ik_required = False

    async def play_move(
        self,
        move: Move,
        play_frequency: float = 100.0,
        initial_goto_duration: float = 0.0,
    ) -> None:
        """Asynchronously play a Move.

        Args:
            move (Move): The Move object to be played.
            play_frequency (float): The frequency at which to evaluate the move (in Hz).
            initial_goto_duration (float): Duration for an initial goto to the move's starting position. If 0.0, no initial goto is performed.

        """
        if not self._try_start_move():
            self.logger.warning("Ignoring play_move request: another move is running.")
            return

        try:
            if initial_goto_duration > 0.0:
                start_head_pose, start_antennas_positions, start_body_yaw = (
                    move.evaluate(0.0)
                )
                await self.goto_target(
                    head=start_head_pose,
                    antennas=start_antennas_positions,
                    duration=initial_goto_duration,
                    body_yaw=start_body_yaw,
                )
            sleep_period = 1.0 / play_frequency

            if move.sound_path is not None and self.audio is not None:
                self.play_sound(str(move.sound_path))

            t0 = time.time()
            while time.time() - t0 < move.duration:
                t = time.time() - t0

                head, antennas, body_yaw = move.evaluate(t)
                if head is not None:
                    self.set_target_head_pose(head)
                if body_yaw is not None:
                    self.set_target_body_yaw(body_yaw)
                if antennas is not None:
                    self.set_target_antenna_joint_positions(antennas)

                elapsed = time.time() - t0 - t
                if elapsed < sleep_period:
                    await asyncio.sleep(sleep_period - elapsed)
                else:
                    await asyncio.sleep(0.001)
        finally:
            if move.sound_path is not None and self.audio is not None:
                # release audio resources after playing the move sound
                self.audio.stop_playing()
            self._end_move()

    async def goto_target(
        self,
        head: Annotated[NDArray[np.float64], (4, 4)] | None = None,  # 4x4 pose matrix
        antennas: Annotated[NDArray[np.float64], (2,)]
        | None = None,  # [right_angle, left_angle] (in rads)
        duration: float = 0.5,  # Duration in seconds for the movement, default is 0.5 seconds.
        method: InterpolationTechnique = InterpolationTechnique.MIN_JERK,  # can be "linear", "minjerk", "ease" or "cartoon", default is "minjerk"
        body_yaw: float | None = 0.0,  # Body yaw angle in radians
    ) -> None:
        """Asynchronously go to a target head pose and/or antennas position using task space interpolation, in "duration" seconds.

        Args:
            head (np.ndarray | None): 4x4 pose matrix representing the target head pose.
            antennas (np.ndarray | list[float] | None): 1D array with two elements representing the angles of the antennas in radians.
            duration (float): Duration of the movement in seconds.
            method (str): Interpolation method to use ("linear", "minjerk", "ease", "cartoon"). Default is "minjerk".
            body_yaw (float | None): Body yaw angle in radians.

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
        | None = None,  # [right_angle, left_angle] length 2
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

            self.set_target_head_joint_positions(head_joint)
            self.set_target_antenna_joint_positions(antennas_joint)
            await asyncio.sleep(0.01)

    def set_recording_publisher(self, publisher: zenoh.Publisher) -> None:
        """Set the publisher for recording data.

        Args:
            publisher: A publisher object that will be used to publish recorded data.

        """
        self.recording_publisher = publisher

    def append_record(self, record: dict[str, Any]) -> None:
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

    def get_present_head_joint_positions(self) -> Annotated[NDArray[np.float64], (7,)]:
        """Return the present head joint positions.

        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The method get_present_head_joint_positions should be overridden by subclasses."
        )

    def get_present_body_yaw(self) -> float:
        """Return the present body yaw."""
        yaw: float = self.get_present_head_joint_positions()[0]
        return yaw

    def get_present_head_pose(self) -> Annotated[NDArray[np.float64], (4, 4)]:
        """Return the present head pose as a 4x4 matrix."""
        assert self.current_head_pose is not None, (
            "The current head pose is not set. Please call the update_head_kinematics_model method first."
        )
        return self.current_head_pose

    def get_current_head_pose(self) -> Annotated[NDArray[np.float64], (4, 4)]:
        """Return the present head pose as a 4x4 matrix."""
        return self.get_present_head_pose()

    def get_present_antenna_joint_positions(
        self,
    ) -> Annotated[NDArray[np.float64], (2,)]:
        """Return the present antenna joint positions.

        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The method get_present_antenna_joint_positions should be overridden by subclasses."
        )

    # Kinematics methods
    def update_head_kinematics_model(
        self,
        head_joint_positions: Annotated[NDArray[np.float64], (7,)] | None = None,
        antennas_joint_positions: Annotated[NDArray[np.float64], (2,)] | None = None,
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

    def set_automatic_body_yaw(self, body_yaw: bool) -> None:
        """Set the automatic body yaw.

        Args:
            body_yaw (bool): The yaw angle of the body.

        """
        self.head_kinematics.set_automatic_body_yaw(automatic_body_yaw=body_yaw)

    def get_urdf(self) -> str:
        """Get the URDF representation of the robot."""
        urdf_path = Path(URDF_ROOT_PATH) / "robot.urdf"

        with open(urdf_path, "r") as f:
            return f.read()

    # Multimedia methods
    def play_sound(self, sound_file: str) -> None:
        """Play a sound file from the assets directory.

        If the file is not found in the assets directory, try to load the path itself.

        Args:
            sound_file (str): The name of the sound file to play (e.g., "wake_up.wav").

        """
        if self.audio:
            self.audio.start_playing()
            self.audio.play_sound(sound_file)

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

    SLEEP_ANTENNAS_JOINT_POSITIONS = np.array((-3.05, 3.05))
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
            antennas=np.array((0.0, 0.0)),
            duration=magic_distance * 20 / 1000,  # ms_per_magic_mm = 10
        )
        await asyncio.sleep(0.1)

        # Toudoum
        self.play_sound("wake_up.wav")

        # Roll 20° to the left
        pose = self.INIT_HEAD_POSE.copy()
        pose[:3, :3] = R.from_euler("xyz", [20, 0, 0], degrees=True).as_matrix()
        await self.goto_target(pose, duration=0.2)

        # Go back to the initial position
        await self.goto_target(self.INIT_HEAD_POSE, duration=0.2)
        if self.audio:
            self.audio.stop_playing()

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
                    self.INIT_HEAD_POSE, antennas=np.array((0.0, 0.0)), duration=1
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
        if self.audio:
            self.audio.stop_playing()

    # Motor control modes
    @abstractmethod
    def get_motor_control_mode(self) -> MotorControlMode:
        """Get the motor control mode."""
        pass

    @abstractmethod
    def set_motor_control_mode(self, mode: MotorControlMode) -> None:
        """Set the motor control mode."""
        pass

    @abstractmethod
    def set_motor_torque_ids(self, ids: list[str], on: bool) -> None:
        """Set the motor torque for specific motor names."""
        pass

    def write_raw_packet(self, packet: bytes) -> bytes:
        """Write a raw packet to the motor controller and return the response.

        Args:
            packet (bytes): The raw packet to send to the motor controller.

        Returns:
            bytes: The raw response packet from the motor controller.

        """
        raise NotImplementedError(
            "The method write_raw_packet is only available for the real robot backend."
        )

    def get_present_passive_joint_positions(self) -> Optional[Dict[str, float]]:
        """Get the present passive joint positions.

        Requires the Placo kinematics engine.
        """
        # This is would be better, and fix mypy issues, but Placo is dynamically imported
        # if not isinstance(self.head_kinematics, PlacoKinematics):
        if self.kinematics_engine != "Placo":
            return None
        return {
            "passive_1_x": self.head_kinematics.get_joint("passive_1_x"),  # type: ignore [union-attr]
            "passive_1_y": self.head_kinematics.get_joint("passive_1_y"),  # type: ignore [union-attr]
            "passive_1_z": self.head_kinematics.get_joint("passive_1_z"),  # type: ignore [union-attr]
            "passive_2_x": self.head_kinematics.get_joint("passive_2_x"),  # type: ignore [union-attr]
            "passive_2_y": self.head_kinematics.get_joint("passive_2_y"),  # type: ignore [union-attr]
            "passive_2_z": self.head_kinematics.get_joint("passive_2_z"),  # type: ignore [union-attr]
            "passive_3_x": self.head_kinematics.get_joint("passive_3_x"),  # type: ignore [union-attr]
            "passive_3_y": self.head_kinematics.get_joint("passive_3_y"),  # type: ignore [union-attr]
            "passive_3_z": self.head_kinematics.get_joint("passive_3_z"),  # type: ignore [union-attr]
            "passive_4_x": self.head_kinematics.get_joint("passive_4_x"),  # type: ignore [union-attr]
            "passive_4_y": self.head_kinematics.get_joint("passive_4_y"),  # type: ignore [union-attr]
            "passive_4_z": self.head_kinematics.get_joint("passive_4_z"),  # type: ignore [union-attr]
            "passive_5_x": self.head_kinematics.get_joint("passive_5_x"),  # type: ignore [union-attr]
            "passive_5_y": self.head_kinematics.get_joint("passive_5_y"),  # type: ignore [union-attr]
            "passive_5_z": self.head_kinematics.get_joint("passive_5_z"),  # type: ignore [union-attr]
            "passive_6_x": self.head_kinematics.get_joint("passive_6_x"),  # type: ignore [union-attr]
            "passive_6_y": self.head_kinematics.get_joint("passive_6_y"),  # type: ignore [union-attr]
            "passive_6_z": self.head_kinematics.get_joint("passive_6_z"),  # type: ignore [union-attr]
            "passive_7_x": self.head_kinematics.get_joint("passive_7_x"),  # type: ignore [union-attr]
            "passive_7_y": self.head_kinematics.get_joint("passive_7_y"),  # type: ignore [union-attr]
            "passive_7_z": self.head_kinematics.get_joint("passive_7_z"),  # type: ignore [union-attr]
        }
