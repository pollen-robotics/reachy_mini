"""Base class for robot backends, simulated or real.

This module defines the `Backend` class, which serves as a base for implementing
different types of robot backends, whether they are simulated (like Mujoco) or real
(connected via serial port). The class provides methods for managing joint positions,
torque control, and other backend-specific functionalities.
It is designed to be extended by subclasses that implement the specific behavior for
each type of backend.
"""

import threading
from typing import List
import numpy as np

class Backend:
    """Base class for robot backends, simulated or real."""

    def __init__(self) -> None:
        """Initialize the backend."""
        self.should_stop = threading.Event()
        self.ready = threading.Event()

        self.head_joint_positions = None  # [yaw, 0, 1, 2, 3, 4, 5]
        self.antenna_joint_positions = None  # [0, 1]
        self.joint_positions_publisher = None  # Placeholder for a publisher object
        self.error = None  # To store any error that occurs during execution
        
        # variables to store the last computed head joint positions and pose
        self._last_head_joint_positions = None  # To store the last head joint positions
        self._last_head_pose = None  # To store the last head pose
        self._kinematics_computation_tolerance = 1e-3  # Tolerance for kinematics computations
        self._last_body_yaw = 0.0  # Last body yaw used in IK computations
        
    def wrapped_run(self):
        """Run the backend in a try-except block to store errors."""
        try:
            self.run()
        except Exception as e:
            self.error = str(e)
            self.close()
            raise e

        self.head_joint_current = None  # Placeholder for head joint torque
        self.head_operation_mode = None  # Placeholder for head operation mode

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

    def set_head_pose(self, pose: np.ndarray, body_yaw: float = 0.0) -> None:
        """Set the head pose. Computes the IK and sets the head joint positions.

        Args:
            pose (np.ndarray): 4x4 pose matrix representing the head pose.
            body_yaw (float): The yaw angle of the body, used to adjust the head pose.

        """

        #check if the pose is the same as the current one
        if self.head_pose is not None and \
            np.allclose(self._last_body_yaw, body_yaw, atol=self._kinematics_computation_tolerance) and \
            np.allclose(self.head_pose, pose, atol=self._kinematics_computation_tolerance):
            # If the pose is the same, do not recompute IK
            return
        

        joints = self.head_kinematics.ik(
            pose, body_yaw=body_yaw, check_collision=self.check_collision
        )
        if joints is None:
            raise ValueError("Could not compute inverse kinematics for the given pose.")

        self.head_pose = pose
        self._last_body_yaw = body_yaw
        
        self.set_head_joint_positions(joints)

    def set_check_collision(self, check: bool) -> None:
        """Set whether to check collisions.

        Args:
            check (bool): If True, the backend will check for collisions.

        """
        self.check_collision = check

    def set_head_joint_positions(self, positions: List[float]) -> None:
        """Set the head joint positions.

        Args:
            positions (List[float]): A list of joint positions for the head.

        """
        self.head_joint_positions = positions

    def set_antenna_joint_positions(self, positions: List[float]) -> None:
        """Set the antenna joint positions.

        Args:
            positions (List[float]): A list of joint positions for the antenna.

        """
        self.antenna_joint_positions = positions

    def set_head_joint_current(self, current: List[int]) -> None:
        """Set the head joint current.

        Args:
            current (List[float]): A list of current values for the head motors.

        """
        self.head_joint_current = current

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

    def get_head_joint_positions(self) -> List[float]:
        """Return the current head joint positions.

        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The method get_head_joint_positions should be overridden by subclasses."
        )

    def get_antenna_joint_positions(self) -> List[float]:
        """Return the current antenna joint positions.

        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The method get_antenna_joint_positions should be overridden by subclasses."
        )

    def get_status(self):
        """Return backend statistics.

        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The method get_status should be overridden by subclasses."
        )
