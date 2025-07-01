import json
import time
from dataclasses import dataclass
from multiprocessing import Event  # It seems to be more accurate than threading.Event
from typing import Optional

import numpy as np
from reachy_mini_motor_controller import ReachyMiniMotorController

from reachy_mini.io.backend import Backend


class RobotBackend(Backend):
    def __init__(self, serialport: str):
        super().__init__()
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

    def run(self):
        assert self.c is not None, "Motor controller not initialized or already closed."

        period = 1.0 / self.control_loop_frequency  # Control loop period in seconds

        self.retries = 5
        self.stats_record_t0 = time.time()

        next_call_event = Event()

        while not self.should_stop.is_set():
            start_t = time.time()
            self.update()
            took = time.time() - start_t

            sleep_time = max(0, period - took)
            if sleep_time > 0:
                next_call_event.clear()
                next_call_event.wait(sleep_time)

    def update(self):
        assert self.c is not None, "Motor controller not initialized or already closed."

        if self._torque_enabled:
            if self.head_joint_positions is not None:
                self.c.set_stewart_platform_position(self.head_joint_positions[1:])
                self.c.set_body_rotation(self.head_joint_positions[0])
            if self.antenna_joint_positions is not None:
                self.c.set_antennas_positions(self.antenna_joint_positions)

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

                # If we never received a position, we retry a few times
                # But most likely the robot is not powered on or connected
                if self.last_alive is None:
                    if self.retries > 0:
                        print(
                            f"Error reading positions, retrying ({self.retries} left): {e}"
                        )
                        self.retries -= 1
                        time.sleep(0.1)
                        return
                    print("No response from the robot, stopping.")
                    print("Make sure the robot is powered on and connected.")
                    self._status.error = "Motors are not powered on or connected."
                    self.should_stop.set()
                    return

                if self.last_alive + 2 < time.time():
                    self._status.error = (
                        "No response from the robot's motor for the last 2 seconds."
                    )

                    print("No response from the robot for 2 seconds, stopping.")
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

    def set_torque(self, enabled: bool) -> None:
        assert self.c is not None, "Motor controller not initialized or already closed."

        if enabled:
            self.c.enable_torque()
        else:
            self.c.disable_torque()

        self._torque_enabled = enabled

    def close(self) -> None:
        self.c = None

    def get_status(self) -> "RobotBackendStatus":
        return self._status


@dataclass
class RobotBackendStatus:
    ready: bool
    last_alive: Optional[float]
    control_loop_stats: dict
    error: Optional[str] = None
