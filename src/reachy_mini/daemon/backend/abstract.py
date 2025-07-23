"""Base class for robot backends, simulated or real.

This module defines the `Backend` class, which serves as a base for implementing
different types of robot backends, whether they are simulated (like Mujoco) or real
(connected via serial port). The class provides methods for managing joint positions,
torque control, and other backend-specific functionalities.
It is designed to be extended by subclasses that implement the specific behavior for
each type of backend.
"""

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

    def __init__(self) -> None:
        """Initialize the backend."""
        self.should_stop = threading.Event()
        self.ready = threading.Event()

        self.head_kinematics = PlacoKinematics(Backend.urdf_root_path)
        self.check_collision = False
        self.gravity_compensation_mode = False  # Flag for gravity compensation mode

        self.head_pose = None  # 4x4 pose matrix
        self.head_joint_positions = None  # [yaw, 0, 1, 2, 3, 4, 5]
        self.antenna_joint_positions = None  # [0, 1]
        self.joint_positions_publisher = None  # Placeholder for a publisher object
        self.pose_publisher = None  # Placeholder for a pose publisher object
        self.error = None  # To store any error that occurs during execution
        
        # variables to store the last computed head joint positions and pose
        self._last_head_joint_positions = None  # To store the last head joint positions
        self._last_head_pose = None  # To store the last head pose
        self._kinematics_computation_tolerance = 1e-3  # Tolerance for kinematics computations        
        self._last_body_yaw = None  # Last body yaw used in IK computations
        
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
            self._last_body_yaw is not None and \
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

    def get_head_pose(
        self, head_joint_positions: List[float] | None = None
    ) -> np.ndarray:
        """Return the current head pose as a 4x4 matrix."""
        if head_joint_positions is None:
            head_joint_positions = self.get_head_joint_positions()

        # filter unnecessary calls to FK
        # check if the head joint positions have changed
        if self._last_head_joint_positions is not None and \
            self._last_head_pose is not None and \
            np.allclose(self._last_head_joint_positions, head_joint_positions, atol=self._kinematics_computation_tolerance):
            # If the head joint positions have not changed, return the cached pose
            return self._last_head_pose
        else:  
            # Compute the forward kinematics to get the current head pose
            self._last_head_pose = self.head_kinematics.fk(head_joint_positions, self.check_collision)

        # Check if the FK was successful
        assert self._last_head_pose is not None, "FK failed to compute the current head pose."

        # Store the last head joint positions
        self._last_head_joint_positions = head_joint_positions

        return self._last_head_pose

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
        head_joints = self.get_head_joint_positions()
        gravity_torque = self.head_kinematics.compute_gravity_torque(head_joints)
        # Convert the torque from Nm to mA
        current = gravity_torque * from_Nm_to_mA / correction_factor
        # Set the head joint current
        self.set_head_joint_current(current)
