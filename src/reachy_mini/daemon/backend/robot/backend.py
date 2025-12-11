"""Robot Backend for Reachy Mini.

This module provides the `RobotBackend` class, which interfaces with the Reachy Mini motor controller to control the robot's movements and manage its status.
It handles the control loop, joint positions, torque enabling/disabling, and provides a status report of the robot's backend.
It uses the `ReachyMiniMotorController` to communicate with the robot's motors.
"""

import json
import logging
import struct
import time
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import Event  # It seems to be more accurate than threading.Event
from typing import Annotated, Any

import log_throttling
import numpy as np
import numpy.typing as npt
import zenoh
from reachy_mini_motor_controller import ReachyMiniPyControlLoop

from reachy_mini.utils.hardware_config.parser import parse_yaml_config

from ..abstract import Backend, MotorControlMode


class RobotBackend(Backend):
    """Real robot backend for Reachy Mini."""

    def __init__(
        self,
        serialport: str,
        log_level: str = "INFO",
        check_collision: bool = False,
        kinematics_engine: str = "AnalyticalKinematics",
        hardware_error_check_frequency: float = 1.0,
        use_audio: bool = True,
        wireless_version: bool = False,
        hardware_config_filepath: str | None = None,
    ):
        """Initialize the RobotBackend.

        Args:
            serialport (str): The serial port to which the Reachy Mini is connected.
            log_level (str): The logging level for the backend. Default is "INFO".
            check_collision (bool): If True, enable collision checking. Default is False.
            kinematics_engine (str): Kinematics engine to use. Defaults to "AnalyticalKinematics".
            hardware_error_check_frequency (float): Frequency in seconds to check for hardware errors. Default is 1.0.
            use_audio (bool): If True, use audio. Default is True.
            wireless_version (bool): If True, indicates that the wireless version of Reachy Mini is used. Default is False.
            hardware_config_filepath (str | None): Path to the hardware configuration YAML file. Default is None.

        Tries to connect to the Reachy Mini motor controller and initializes the control loop.

        """
        super().__init__(
            check_collision=check_collision,
            kinematics_engine=kinematics_engine,
            use_audio=use_audio,
            wireless_version=wireless_version,
        )

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        self.control_loop_frequency = 50.0  # Hz
        self.c: ReachyMiniPyControlLoop | None = ReachyMiniPyControlLoop(
            serialport,
            read_position_loop_period=timedelta(
                seconds=1.0 / self.control_loop_frequency
            ),
            allowed_retries=5,
            stats_pub_period=timedelta(seconds=1.0),
        )

        self.name2id = self.c.get_motor_name_id()
        if hardware_config_filepath is not None:
            config = parse_yaml_config(hardware_config_filepath)
            for motor_name, motor_conf in config.motors.items():
                if motor_conf.pid is not None:
                    motor_id = self.name2id[motor_name]
                    p, i, d = motor_conf.pid
                    self.logger.info(
                        f"Setting PID gains for motor '{motor_name}' (ID: {motor_id}): P={p}, I={i}, D={d}"
                    )
                    self.c.async_write_pid_gains(motor_id, p, i, d)

        self.motor_control_mode = self._infer_control_mode()
        self._torque_enabled = self.motor_control_mode != MotorControlMode.Disabled
        self.logger.info(f"Motor control mode: {self.motor_control_mode}")
        self.last_alive: float | None = None

        self._status = RobotBackendStatus(
            motor_control_mode=self.motor_control_mode,
            ready=False,
            last_alive=None,
            control_loop_stats={},
        )
        self._stats_record_period = 1.0  # seconds
        self._stats: dict[str, Any] = {
            "timestamps": [],
            "nb_error": 0,
            "record_period": self._stats_record_period,
        }

        self._current_head_operation_mode = -1  # Default to torque control mode
        self._current_antennas_operation_mode = -1  # Default to torque control mode
        self.target_antenna_joint_current = None  # Placeholder for antenna joint torque
        self.target_head_joint_current = None  # Placeholder for head joint torque

        if hardware_error_check_frequency <= 0:
            raise ValueError(
                "hardware_error_check_frequency must be positive and non-zero (Hz)."
            )

        self.hardware_error_check_period = (
            1.0 / hardware_error_check_frequency
        )  # seconds

        # Initialize IMU for wireless version
        if wireless_version:
            try:
                from bmi088 import BMI088

                self.bmi088 = BMI088(i2c_bus=4)
                self.logger.info("BMI088 IMU initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize IMU: {e}")
                self.bmi088 = None
        else:
            self.bmi088 = None

        self.imu_publisher: zenoh.Publisher | None = None

    def run(self) -> None:
        """Run the control loop for the robot backend.

        This method continuously updates the motor controller at a specified frequency.
        It reads the joint positions, updates the motor controller, and publishes the joint positions.
        It also handles errors and retries if the motor controller is not responding.
        """
        assert self.c is not None, "Motor controller not initialized or already closed."

        period = 1.0 / self.control_loop_frequency  # Control loop period in seconds

        self.retries = 5
        self.stats_record_t0 = time.time()

        self.last_hardware_error_check_time = time.time()

        next_call_event = Event()

        # Compute the forward kinematics to get the initial head pose
        # IMPORTANT for wake_up
        head_positions, _ = self.get_all_joint_positions()
        # make sure to converge fully (a lot of iterations)
        self.current_head_pose = self.head_kinematics.fk(
            np.array(head_positions),
            no_iterations=20,
        )
        assert self.current_head_pose is not None

        self.head_kinematics.ik(self.current_head_pose, no_iterations=20)

        while not self.should_stop.is_set():
            start_t = time.time()
            self._stats["timestamps"].append(time.time())
            self._update()
            took = time.time() - start_t

            sleep_time = period - took
            if sleep_time < 0:
                self.logger.debug(
                    f"Control loop took too long: {took * 1000:.3f} ms, expected {period * 1000:.3f} ms"
                )
                sleep_time = 0.001

            next_call_event.clear()
            next_call_event.wait(sleep_time)

    def _update(self) -> None:
        assert self.c is not None, "Motor controller not initialized or already closed."

        if self._torque_enabled:
            if self._current_head_operation_mode != 0:  # if position control mode
                if self.target_head_joint_positions is not None:
                    self.c.set_stewart_platform_position(
                        self.target_head_joint_positions[1:].tolist()
                    )
                    self.c.set_body_rotation(self.target_head_joint_positions[0])
            else:  # it's in torque control mode
                if self.gravity_compensation_mode:
                    # This function will set the head_joint_current
                    # to the current necessary to compensate for gravity
                    self.compensate_head_gravity()
                if self.target_head_joint_current is not None:
                    self.c.set_stewart_platform_goal_current(
                        np.round(self.target_head_joint_current[1:], 0)
                        .astype(int)
                        .tolist()
                    )
                    # Body rotation torque control is not supported with feetech motors
                    # self.c.set_body_rotation_goal_current(int(self.target_head_joint_current[0]))

            if self._current_antennas_operation_mode != 0:  # if position control mode
                if self.target_antenna_joint_positions is not None:
                    self.c.set_antennas_positions(
                        self.target_antenna_joint_positions.tolist()
                    )
            # Antenna torque control is not supported with feetech motors
            # else:
            #     if self.target_antenna_joint_current is not None:
            #         self.c.set_antennas_goal_current(
            #            np.round(self.target_antenna_joint_current, 0).astype(int).tolist()
            #         )

        if (
            self.joint_positions_publisher is not None
            and self.pose_publisher is not None
        ):
            try:
                head_positions, antenna_positions = self.get_all_joint_positions()

                # Update the head kinematics model with the current head positions
                self.update_head_kinematics_model(
                    np.array(head_positions),
                    np.array(antenna_positions),
                )

                # Update the target head joint positions from IK if necessary
                # - does nothing if the targets did not change
                if self.ik_required:
                    try:
                        self.update_target_head_joints_from_ik(
                            self.target_head_pose, self.target_body_yaw
                        )
                    except ValueError as e:
                        log_throttling.by_time(self.logger, interval=0.5).warning(
                            f"IK error: {e}"
                        )

                if not self.is_shutting_down:
                    self.joint_positions_publisher.put(
                        json.dumps(
                            {
                                "head_joint_positions": head_positions,
                                "antennas_joint_positions": antenna_positions,
                            }
                        )
                    )
                    self.pose_publisher.put(
                        json.dumps(
                            {
                                "head_pose": self.get_present_head_pose().tolist(),
                            }
                        )
                    )

                    # Publish IMU data if available
                    if self.imu_publisher is not None and self.bmi088 is not None:
                        imu_data = self.get_imu_data()
                        if imu_data is not None:
                            self.imu_publisher.put(json.dumps(imu_data))

                self.last_alive = time.time()

                self.ready.set()  # Mark the backend as ready
            except RuntimeError as e:
                self._stats["nb_error"] += 1

                assert self.last_alive is not None

                if self.last_alive + 1 < time.time():
                    self.error = (
                        "No response from the robot's motor for the last second."
                    )

                    self.logger.error(
                        "No response from the robot for the last second, stopping."
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
                    self._status.control_loop_stats["motor_controller"] = str(
                        self.c.get_stats()
                    )

                self._stats["timestamps"].clear()
                self._stats["nb_error"] = 0
                self.stats_record_t0 = time.time()

            if (
                time.time() - self.last_hardware_error_check_time
                > self.hardware_error_check_period
            ):
                hardware_errors = self.read_hardware_errors()
                if hardware_errors:
                    for motor_name, errors in hardware_errors.items():
                        self.logger.error(
                            f"Motor '{motor_name}' hardware errors: {errors}"
                        )
                self.last_hardware_error_check_time = time.time()

    def close(self) -> None:
        """Close the motor controller connection and release resources."""
        if self.c is not None:
            self.c.close()
        self.c = None
        super().close()

    def get_status(self) -> "RobotBackendStatus":
        """Get the current status of the robot backend."""
        self._status.error = self.error
        self._status.motor_control_mode = self.motor_control_mode
        return self._status

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
            f" Got {mode} instead"
        )

        # if motors are enabled, disable them before changing the mode
        if self._torque_enabled:
            self.c.enable_stewart_platform(False)
        # set the new operation mode
        self.c.set_stewart_platform_operating_mode(mode)

        if mode != 0:
            # if the mode is not torque control, we need to set the head joint positions
            # to the current positions to avoid sudden movements
            motor_pos = self.c.get_last_position()
            self.target_head_joint_positions = np.array(
                [motor_pos.body_yaw] + motor_pos.stewart
            )

            self.c.set_stewart_platform_position(
                self.target_head_joint_positions[1:].tolist()
            )
            self.c.set_body_rotation(self.target_head_joint_positions[0])
            self.c.enable_body_rotation(True)
            self.c.set_body_rotation_operating_mode(0)
        else:
            self.c.enable_body_rotation(False)

        if self._torque_enabled:
            self.c.enable_stewart_platform(True)

        self._current_head_operation_mode = mode

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
        assert mode in [
            0,
            3,
            5,
        ], (
            "Invalid operation mode. Must be one of [0 (torque), 3 (position), 5 (current-limiting position)]."
        )

        if self._current_antennas_operation_mode != mode:
            if mode != 0:
                # if the mode is not torque control, we need to set the head joint positions
                # to the current positions to avoid sudden movements
                self.target_antenna_joint_positions = np.array(
                    self.c.get_last_position().antennas
                )
                self.c.set_antennas_positions(
                    self.target_antenna_joint_positions.tolist()
                )
                self.c.enable_antennas(True)
            else:
                self.c.enable_antennas(False)

            self._current_antennas_operation_mode = mode

    def get_all_joint_positions(self) -> tuple[list[float], list[float]]:
        """Get the current joint positions of the robot.

        Returns:
            tuple: A tuple containing two lists - the first list is for the head joint positions,
                    and the second list is for the antenna joint positions.

        """
        assert self.c is not None, "Motor controller not initialized or already closed."
        positions = self.c.get_last_position()

        yaw = positions.body_yaw
        antennas = positions.antennas
        dofs = positions.stewart

        return [yaw] + list(dofs), list(antennas)

    def get_present_head_joint_positions(
        self,
    ) -> Annotated[npt.NDArray[np.float64], (7,)]:
        """Get the current joint positions of the head.

        Returns:
            list: A list of joint positions for the head, including the body rotation.

        """
        return np.array(self.get_all_joint_positions()[0])

    def get_present_antenna_joint_positions(
        self,
    ) -> Annotated[npt.NDArray[np.float64], (2,)]:
        """Get the current joint positions of the antennas.

        Returns:
            list: A list of joint positions for the antennas.

        """
        return np.array(self.get_all_joint_positions()[1])

    def get_imu_data(self) -> dict[str, list[float] | float] | None:
        """Get current IMU data (accelerometer, gyroscope, quaternion, temperature).

        Returns:
            dict with 'accelerometer', 'gyroscope', 'quaternion', and 'temperature' keys,
            or None if IMU is not available.

        """
        if self.bmi088 is None:
            return None

        try:
            # Read accelerometer (returns tuple of x, y, z in m/s^2)
            accel_x, accel_y, accel_z = self.bmi088.read_accelerometer(m_per_s2=True)

            # Read gyroscope (returns tuple of x, y, z in rad/s)
            gyro_x, gyro_y, gyro_z = self.bmi088.read_gyroscope(deg_per_s=False)

            # Get quaternion orientation (dt = control loop period)
            dt = 1.0 / self.control_loop_frequency  # 0.02 seconds at 50Hz
            quat = self.bmi088.get_quat(dt)

            # Read temperature in Celsius
            temperature = self.bmi088.read_temperature()

            # Convert all numpy types to native Python floats for JSON serialization
            return {
                "accelerometer": [float(accel_x), float(accel_y), float(accel_z)],
                "gyroscope": [float(gyro_x), float(gyro_y), float(gyro_z)],
                "quaternion": [float(q) for q in quat],
                "temperature": float(temperature),
            }
        except Exception as e:
            self.logger.error(f"Error reading IMU data: {e}")
            return None

    def compensate_head_gravity(self) -> None:
        """Calculate the currents necessary to compensate for gravity."""
        assert self.kinematics_engine == "Placo", (
            "Gravity compensation is only supported with the Placo kinematics engine."
        )

        # Even though in their docs dynamixes says that 1 count is 1 mA, in practice I've found it to be 3mA.
        # I am not sure why this happens
        # Another explanation is that our model is bad and the current is overestimated 3x (but I have not had these issues with other robots)
        # So I am using a magic number to compensate for this.
        # for currents under 30mA the constant is around 1
        from_Nm_to_mA = 1.47 / 0.52 * 1000
        # Conversion factor from Nm to mA for the Stewart platform motors
        # The torque constant is not linear, so we need to use a correction factor
        # This is a magic number that should be determined experimentally
        # For currents under 30mA, the constant is around 4.0
        # Then it drops to 1.0 for currents above 1.5A
        correction_factor = 4.0
        # Get the current head joint positions
        head_joints = self.get_present_head_joint_positions()
        gravity_torque = self.head_kinematics.compute_gravity_torque(  # type: ignore [union-attr]
            np.array(head_joints)
        )
        # Convert the torque from Nm to mA
        current = gravity_torque * from_Nm_to_mA / correction_factor
        # Set the head joint current
        self.set_target_head_joint_current(current)

    def get_motor_control_mode(self) -> MotorControlMode:
        """Get the motor control mode."""
        return self.motor_control_mode

    def set_motor_control_mode(self, mode: MotorControlMode) -> None:
        """Set the motor control mode."""
        # Check if the mode is already set
        if mode == self.motor_control_mode:
            return

        if mode == MotorControlMode.Enabled:
            if self.motor_control_mode == MotorControlMode.GravityCompensation:
                # First, make sure we switch to position control
                self.disable_motors()
                self.set_head_operation_mode(3)
                self.set_antennas_operation_mode(3)

            self.gravity_compensation_mode = False
            self.enable_motors()

        elif mode == MotorControlMode.Disabled:
            self.gravity_compensation_mode = False
            self.disable_motors()

        elif mode == MotorControlMode.GravityCompensation:
            if self.kinematics_engine != "Placo":
                raise RuntimeError(
                    "Gravity compensation mode is only supported with the Placo kinematics engine."
                )

            self.disable_motors()
            self.set_head_operation_mode(0)
            self.set_antennas_operation_mode(0)
            self.gravity_compensation_mode = True
            self.enable_motors()

        self.motor_control_mode = mode

    def set_motor_torque_ids(self, ids: list[str], on: bool) -> None:
        """Set the torque state for specific motor names.

        Args:
            ids (list[int]): List of motor IDs to set the torque state for.
            on (bool): True to enable torque, False to disable.

        """
        assert self.c is not None, "Motor controller not initialized or already closed."

        assert ids is not None and len(ids) > 0, "IDs list cannot be empty or None."

        ids_int = [self.name2id[name] for name in ids]

        if on:
            self.c.enable_torque_on_ids(ids_int)
        else:
            self.c.disable_torque_on_ids(ids_int)

    def _infer_control_mode(self) -> MotorControlMode:
        assert self.c is not None, "Motor controller not initialized or already closed."

        torque = self.c.is_torque_enabled()

        if not torque:
            return MotorControlMode.Disabled

        mode = self.c.get_stewart_platform_operating_mode()
        if mode == 3:
            return MotorControlMode.Enabled
        elif mode == 1:
            return MotorControlMode.GravityCompensation
        else:
            raise ValueError(f"Unknown motor control mode: {mode}")

    def read_hardware_errors(self) -> dict[str, list[str]]:
        """Read hardware errors from the motor controller."""
        if self.c is None:
            return {}

        def decode_hardware_error_byte(err_byte: int) -> list[str]:
            # https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/#hardware-error-status
            bits_to_error = {
                0: "Input Voltage Error",
                2: "Overheating Error",
                4: "Electrical Shock Error",
                5: "Overload Error",
            }
            err_bits = [i for i in range(8) if (err_byte & (1 << i)) != 0]
            return [bits_to_error[b] for b in err_bits if b in bits_to_error]

        def voltage_ok(
            id: int,
            allowed_max_voltage: float = 7.8,
        ) -> bool:
            assert self.c is not None, (
                "Motor controller not initialized or already closed."
            )
            # https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/#present-input-voltage
            resp_bytes = self.c.async_read_raw_bytes(id, 144, 2)
            resp = struct.unpack("h", bytes(resp_bytes))[0]
            voltage: float = resp / 10.0  # in Volts

            return voltage <= allowed_max_voltage

        errors = {}
        for name, id in self.c.get_motor_name_id().items():
            # https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/#hardware-error-status
            err_byte = self.c.async_read_raw_bytes(id, 70, 1)
            assert len(err_byte) == 1
            err = decode_hardware_error_byte(err_byte[0])
            if err:
                if "Input Voltage Error" in err:
                    if voltage_ok(id):
                        err.remove("Input Voltage Error")

                # To avoid logging empty errors like "Motor 1: []"
                if len(err) > 0:
                    errors[name] = err

        return errors

    def write_raw_packet(self, packet: bytes) -> bytes:
        """Write a raw packet to the motor controller and return the response.

        Args:
            packet (bytes): The raw packet to send to the motor controller.

        Returns:
            bytes: The raw response packet from the motor controller.

        """
        assert self.c is not None, "Motor controller not initialized or already closed."

        result: bytes = bytes(self.c.write_raw_packet(packet))
        return result


@dataclass
class RobotBackendStatus:
    """Status of the Robot Backend."""

    ready: bool
    motor_control_mode: MotorControlMode
    last_alive: float | None
    control_loop_stats: dict[str, Any]
    error: str | None = None
