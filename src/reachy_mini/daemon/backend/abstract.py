"""Base class for robot backends, simulated or real.

This module defines the `Backend` class, which serves as a base for implementing
different types of robot backends, whether they are simulated (like Mujoco) or real
(connected via serial port). The class provides methods for managing joint positions,
torque control, and other backend-specific functionalities.
It is designed to be extended by subclasses that implement the specific behavior for
each type of backend.
"""

import json
import logging
import threading
from importlib.resources import files
from typing import List

import numpy as np

import reachy_mini
from reachy_mini.placo_kinematics import PlacoKinematics


class Backend:
    """Base class for robot backends, simulated or real."""

    urdf_root_path: str = str(
        files(reachy_mini).joinpath("descriptions/reachy_mini/urdf")
    )

    def __init__(self, log_level: str = "INFO") -> None:
        """Initialize the backend."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        self.should_stop = threading.Event()
        self.ready = threading.Event()

        self.head_kinematics = PlacoKinematics(Backend.urdf_root_path)
        self.check_collision = False
        self.gravity_compensation_mode = False  # Flag for gravity compensation mode

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
        self._last_collision_check = None  # Track the last collision flag used in IK
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

    def update_target_head_joints_from_ik(self, pose: np.ndarray = None, body_yaw: float = None) -> None:
        """Update the target head joint positions from inverse kinematics.

        Args:
            pose (np.ndarray): 4x4 pose matrix representing the head pose.
            body_yaw (float): The yaw angle of the body, used to adjust the head pose.

        """
        if pose is None:
            pose = self.target_head_pose if self.target_head_pose is not None else np.eye(4)

        if body_yaw is None:
            body_yaw = self.target_body_yaw if self.target_body_yaw is not None else 0.0

        # Compute the inverse kinematics to get the head joint positions
        joints = self.head_kinematics.ik(
            pose, body_yaw=body_yaw, check_collision=self.check_collision
        )

        if joints is None or np.any(np.isnan(joints)):
            raise ValueError("WARNING: Collision detected or head pose not achievable!")

        # update the target head pose and body yaw
        self._last_target_head_pose = pose
        self._last_target_body_yaw = body_yaw
        self._last_collision_check = self.check_collision

        self.set_target_head_joint_positions(joints)

    def set_target_head_pose(self, pose: np.ndarray, body_yaw: float = 0.0) -> None:
        """Set the target head pose for the robot.

        Args:
            pose (np.ndarray): 4x4 pose matrix representing the head pose.
            body_yaw (float): The yaw angle of the body, used to adjust the head pose.

        """
        # update the target head pose and body yaw
        self.target_head_pose = pose
        self.target_body_yaw = body_yaw
        self.ik_required = True

    def set_target_head_joint_positions(self, positions: List[float]) -> None:
        """Set the head joint positions.

        Args:
            positions (List[float]): A list of joint positions for the head.

        """
        self.target_head_joint_positions = positions

    def set_target_antenna_joint_positions(self, positions: List[float]) -> None:
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

    def set_head_operation_mode(self, mode: int) -> None:
        """Set mode of operation for the head."""
        raise NotImplementedError(
            "The method set_head_operation_mode should be overridden by subclasses."
        )

    def set_antennas_operation_mode(self, mode: int) -> None:
        """Set mode of operation for the antennas."""
        raise NotImplementedError(
            "The method set_antennas_operation_mode should be overridden by subclasses."
        )

    def enable_motors(self) -> None:
        """Enable the motors."""
        raise NotImplementedError(
            "The method enable_motors should be overridden by subclasses."
        )

    def disable_motors(self) -> None:
        """Disable the motors."""
        raise NotImplementedError(
            "The method disable_motors should be overridden by subclasses."
        )

    def get_present_head_joint_positions(self) -> List[float]:
        """Return the present head joint positions.

        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The method get_present_head_joint_positions should be overridden by subclasses."
        )

    def get_present_head_pose(self) -> np.ndarray:
        """Return the present head pose as a 4x4 matrix."""
        assert self.current_head_pose is not None, (
            "The current head pose is not set. Please call the update_head_kinematics_model method first."
        )
        return self.current_head_pose

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

    def set_check_collision(self, check: bool) -> None:
        """Set whether to check collisions.

        Args:
            check (bool): If True, the backend will check for collisions.

        """
        self.check_collision = check

    def set_automatic_body_yaw(self, body_yaw: float) -> None:
        """Set the automatic body yaw.

        Args:
            body_yaw (float): The yaw angle of the body.

        """
        self.head_kinematics.start_body_yaw = body_yaw

    def set_gravity_compensation_mode(self, mode: bool) -> None:
        """Set the gravity compensation mode.

        Args:
            mode (bool): If True, gravity compensation is enabled.

        """
        self.gravity_compensation_mode = mode  # True (enable) or False (disable)

    def compensate_head_gravity(self) -> None:
        """Calculate the currents necessary to compensate for gravity."""
        # Even though in their docs dynamixes says that 1 count is 1 mA, in practice I've found it to be 3mA.
        # I am not sure why this happens
        # Another explanation is that our model is bad and the current is overestimated 3x (but I have not had these issues with other robots)
        # So I am using a magic number to compensate for this.
        # for currents under 30mA the constant is around 1
        from_Nm_to_mA = 1.47 / 0.52 * 1000
        # Conversion factor from Nm to mA for the Stewart platform motors
        # The torque constant is not linear, so we need to use a correction factor
        # This is a magic number that should be determined experimentally
        # For currents under 30mA, the constant is around 3
        # Then it drops to 1.0 for currents above 1.5A
        correction_factor = 3.0
        # Get the current head joint positions
        head_joints = self.get_present_head_joint_positions()
        gravity_torque = self.head_kinematics.compute_gravity_torque(
            np.array(head_joints)
        )
        # Convert the torque from Nm to mA
        current = gravity_torque * from_Nm_to_mA / correction_factor
        # Set the head joint current
        self.set_target_head_joint_current(current.tolist())
