"""Base class for robot backends, simulated or real.

This module defines the `Backend` class, which serves as a base for implementing
different types of robot backends, whether they are simulated (like Mujoco) or real
(connected via serial port). The class provides methods for managing joint positions,
torque control, and other backend-specific functionalities.
It is designed to be extended by subclasses that implement the specific behavior for
each type of backend.
"""

import json
import threading
from typing import List


class Backend:
    """Base class for robot backends, simulated or real."""

    def __init__(self) -> None:
        """Initialize the backend."""
        self.should_stop = threading.Event()
        self.ready = threading.Event()

        self.head_joint_positions = None  # [yaw, 0, 1, 2, 3, 4, 5]
        self.antenna_joint_positions = None  # [0, 1]
        self.joint_positions_publisher = None  # Placeholder for a publisher object
        self.recording_publisher = None  # Placeholder for a recording publisher object
        self.error = None  # To store any error that occurs during execution
        self.is_recording = False  # Flag to indicate if recording is active
        self.recorded_data = []  # List to store recorded data

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

    def set_recording_publisher(self, publisher) -> None:
        """Set the publisher for recording data.

        Args:
            publisher: A publisher object that will be used to publish recorded data.

        """
        self.recording_publisher = publisher

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

    def append_record(self, record: dict) -> None:
        """Append a record to the recorded data.

        Args:
            record (dict): A dictionary containing the record data to be appended.

        """
        if self.is_recording:
            self.recorded_data.append(record)

    def start_recording(self) -> None:
        """Start recording data."""
        self.is_recording = True
        self.recorded_data = []

    def stop_recording(self) -> None:
        """Stop recording data and publish the recorded data."""
        self.is_recording = False
        recorded_data = self.recorded_data.copy()
        self.recording_publisher.put(json.dumps(recorded_data))
        self.recorded_data.clear()

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
