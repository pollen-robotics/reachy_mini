"""Robot Backend for Reachy Mini.

This module provides the `RobotBackend` class, which interfaces with the Reachy Mini motor controller to control the robot's movements and manage its status.
It handles the control loop, joint positions, torque enabling/disabling, and provides a status report of the robot's backend.
It uses the `ReachyMiniMotorController` to communicate with the robot's motors.
"""

import json
import logging
import time
from dataclasses import dataclass
from multiprocessing import Event  # It seems to be more accurate than threading.Event
from typing import Optional

import numpy as np
from reachy_mini_motor_controller import ReachyMiniMotorController

from reachy_mini.daemon.backend.abstract import Backend

logger = logging.getLogger(__name__)


class RobotBackend(Backend):
    """Real robot backend for Reachy Mini."""

    def __init__(self, serialport: str, log_level: str = "INFO"):
        """Initialize the RobotBackend.

        Args:
            serialport (str): The serial port to which the Reachy Mini is connected.
            log_level (str): The logging level for the backend. Default is "INFO".

        Tries to connect to the Reachy Mini motor controller and initializes the control loop.

        """
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        self.c = ReachyMiniMotorController(serialport)
        self.control_loop_frequency = 200.0
        self.last_alive = None

        self._torque_enabled = False

        self._status = RobotBackendStatus(
            ready=False,
            last_alive=None,
            control_loop_stats={},
        )
        self._stats_record_period = 1.0  # seconds
        self._stats = {
            "timestamps": [],
            "nb_error": 0,
            "record_period": self._stats_record_period,
        }
        self._head_operation_mode = -1  # Default to torque control mode
        self._antennas_operation_mode = -1  # Default to torque control mode
        self.antenna_joint_current = None  # Placeholder for antenna joint torque
        self.head_joint_current = None  # Placeholder for head joint torque

    def run(self):
        """Run the control loop for the robot backend.

        This method continuously updates the motor controller at a specified frequency.
        It reads the joint positions, updates the motor controller, and publishes the joint positions.
        It also handles errors and retries if the motor controller is not responding.
        """
        assert self.c is not None, "Motor controller not initialized or already closed."

        period = 1.0 / self.control_loop_frequency  # Control loop period in seconds

        self.retries = 5
        self.stats_record_t0 = time.time()

        next_call_event = Event()

        while not self.should_stop.is_set():
            start_t = time.time()
            self._update()
            took = time.time() - start_t

            sleep_time = max(0, period - took)
            if sleep_time > 0:
                next_call_event.clear()
                next_call_event.wait(sleep_time)

    def _update(self):
        assert self.c is not None, "Motor controller not initialized or already closed."

        if self._torque_enabled:
            if self._head_operation_mode != 0:  # if position control mode
                if self.head_joint_positions is not None:
                    self.c.set_stewart_platform_position(self.head_joint_positions[1:])
                    self.c.set_body_rotation(self.head_joint_positions[0])
            else:  # it's in torque control mode
                if self.head_joint_current is not None:
                    self.c.set_stewart_platform_goal_current(
                        np.round(self.head_joint_current[1:], 0).astype(int).tolist()
                    )
                    # Body rotation torque control is not supported with feetech motors
                    # self.c.set_body_rotation_goal_current(int(self.head_joint_current[0]))

            if self._antennas_operation_mode != 0:  # if position control mode
                if self.antenna_joint_positions is not None:
                    self.c.set_antennas_positions(self.antenna_joint_positions)
            # Antenna torque control is not supported with feetech motors
            # else:
            #     if self.antenna_joint_current is not None:
            #         self.c.set_antennas_goal_current(
            #            np.round(self.antenna_joint_current, 0).astype(int).tolist()
            #         )

        if self.joint_positions_publisher is not None:
            try:
                positions = self.c.read_all_positions()
                yaw = positions[0]
                antennas = positions[1:3]
                dofs = positions[3:]

                self.joint_positions_publisher.put(
                    json.dumps(
                        {
                            "head_joint_positions": [yaw] + list(dofs),
                            "antennas_joint_positions": list(antennas),
                        }
                    )
                )
                self.last_alive = time.time()
                self._stats["timestamps"].append(self.last_alive)

                self.ready.set()  # Mark the backend as ready
            except RuntimeError as e:
                self._stats["nb_error"] += 1
                self.logger.warning(f"Error reading positions: {e}")

                # If we never received a position, we retry a few times
                # But most likely the robot is not powered on or connected
                if self.last_alive is None:
                    if self.retries > 0:
                        self.logger.error(
                            f"Error reading positions, retrying ({self.retries} left): {e}"
                        )
                        self.retries -= 1
                        time.sleep(0.1)
                        return
                    self.logger.error("No response from the robot, stopping.")
                    self.logger.error(
                        "Make sure the robot is powered on and connected."
                    )
                    self.error = "Motors are not powered on or connected."
                    self.should_stop.set()
                    return

                if self.last_alive + 2 < time.time():
                    self.error = (
                        "No response from the robot's motor for the last 2 seconds."
                    )

                    self.logger.error(
                        "No response from the robot for 2 seconds, stopping."
                    )
                    raise e

            if time.time() - self.stats_record_t0 > self._stats_record_period:
                dt = np.diff(self._stats["timestamps"])
                if len(dt) > 1:
                    self._status.control_loop_stats["mean_control_loop_frequency"] = (
                        float(np.mean(1.0 / dt))
                    )
                    self._status.control_loop_stats["max_control_loop_interval"] = (
                        float(np.max(dt))
                    )
                    self._status.control_loop_stats["nb_error"] = self._stats[
                        "nb_error"
                    ]

                self._stats["timestamps"].clear()
                self._stats["nb_error"] = 0
                self.stats_record_t0 = time.time()

    def enable_motors(self) -> None:
        """Enable the motors by turning the torque on."""
        assert self.c is not None, "Motor controller not initialized or already closed."

        self.c.enable_torque()
        self._torque_enabled = True

    def disable_motors(self) -> None:
        """Disable the motors by turning the torque off."""
        assert self.c is not None, "Motor controller not initialized or already closed."

        self.c.disable_torque()
        self._torque_enabled = False

    def set_head_operation_mode(self, mode: int) -> None:
        """Change the operation mode of the head motors.

        Args:
            mode (int): The operation mode for the head motors.

        The operation modes can be:
            0: torque control
            3: position control
            5: current-based position control.

        Important:
            This method does not work well with the current feetech motors (body rotation), as they do not support torque control.
            So the method disables the antennas when in torque control mode.
            The dynamixel motors used for the head do support torque control, so this method works as expected.

        Args:
            mode (int): The operation mode for the head motors.
                        This could be a specific mode like position control, velocity control, or torque control.

        """
        assert self.c is not None, "Motor controller not initialized or already closed."
        assert mode in [0, 3, 5], (
            "Invalid operation mode. Must be one of [0 (torque), 3 (position), 5 (current-limiting position)]."
        )

        if self._head_operation_mode != mode:
            # if motors are enabled, disable them before changing the mode
            if self._torque_enabled:
                self.c.enable_stewart_platform(False)
            # set the new operation mode
            self.c.set_stewart_platform_operating_mode(mode)

            if mode != 0:
                # if the mode is not torque control, we need to set the head joint positions
                # to the current positions to avoid sudden movements
                motor_pos = self.c.read_all_positions()
                self.head_joint_positions = [motor_pos[0]] + motor_pos[3:]
                self.c.set_stewart_platform_position(self.head_joint_positions[1:])
                self.c.set_body_rotation(self.head_joint_positions[0])
                self.c.enable_body_rotation(True)
                self.c.set_body_rotation_operating_mode(0)
            else:
                self.c.enable_body_rotation(False)

            if self._torque_enabled:
                self.c.enable_stewart_platform(True)

            self._head_operation_mode = mode

    def set_antennas_operation_mode(self, mode: int) -> None:
        """Change the operation mode of the antennas motors.

        Args:
            mode (int): The operation mode for the antennas motors (0: torque control, 3: position control, 5: current-based position control).

        Important:
            This method does not work well with the current feetech motors, as they do not support torque control.
            So the method disables the antennas when in torque control mode.

        Args:
            mode (int): The operation mode for the antennas motors.
                        This could be a specific mode like position control, velocity control, or torque control.

        """
        assert self.c is not None, "Motor controller not initialized or already closed."
        assert mode in [0, 3, 5], (
            "Invalid operation mode. Must be one of [0 (torque), 3 (position), 5 (current-limiting position)]."
        )

        if self._antennas_operation_mode != mode:
            if mode != 0:
                # if the mode is not torque control, we need to set the head joint positions
                # to the current positions to avoid sudden movements
                self.antenna_joint_positions = self.c.read_all_positions()[1:3]
                self.c.set_antennas_positions(self.antenna_joint_positions)
                self.c.enable_antennas(True)
            else:
                self.c.enable_antennas(False)

            self._antennas_operation_mode = mode

    def close(self) -> None:
        """Close the motor controller connection."""
        self.c = None

    def get_status(self) -> "RobotBackendStatus":
        """Get the current status of the robot backend."""
        self._status.error = self.error
        return self._status


@dataclass
class RobotBackendStatus:
    """Status of the Robot Backend."""

    ready: bool
    last_alive: Optional[float]
    control_loop_stats: dict
    error: Optional[str] = None
