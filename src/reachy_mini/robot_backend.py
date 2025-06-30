import json
import time

import numpy as np
from reachy_mini_motor_controller import ReachyMiniMotorController

from reachy_mini.io import Backend


class RobotBackend(Backend):
    def __init__(self, serialport: str):
        super().__init__()
        self.c = ReachyMiniMotorController(serialport)
        self.control_loop_frequency = 200.0
        self.publish_frequency = 100.0
        self.decimation = int(self.control_loop_frequency / self.publish_frequency)
        self.last_alive = None

        self._torque_enabled = False

        self._stats_record_period = 1.0  # seconds
        self._stats = {
            "timestamps": [],
            "nb_error": 0,
            "record_period": self._stats_record_period,
        }

    def run(self):
        assert self.c is not None, "Motor controller not initialized or already closed."

        period = 1.0 / self.control_loop_frequency  # Control loop period in seconds
        step = 0
        retries = 5

        stats_record_t0 = time.time()

        while not self.should_stop.is_set():
            start_t = time.time()

            if self._torque_enabled:
                if self.head_joint_positions is not None:
                    self.c.set_stewart_platform_position(self.head_joint_positions[1:])
                    self.c.set_body_rotation(self.head_joint_positions[0])
                if self.antenna_joint_positions is not None:
                    self.c.set_antennas_positions(self.antenna_joint_positions)

            if step % self.decimation == 0:
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
                            if retries > 0:
                                print(
                                    f"Error reading positions, retrying ({retries} left): {e}"
                                )
                                retries -= 1
                                time.sleep(0.1)
                                continue
                            print("No response from the robot, stopping.")
                            print("Make sure the robot is powered on and connected.")
                            break

                        if self.last_alive + 2 < time.time():
                            print("No response from the robot for 2 seconds, stopping.")
                            raise e

            step += 1

            if time.time() - stats_record_t0 > self._stats_record_period:
                dt = np.diff(self._stats["timestamps"])
                if len(dt) > 1:
                    self._last_stats = {
                        "mean_control_loop_frequency": float(np.mean(1.0 / dt)),
                        "max_control_loop_interval": float(np.max(dt)),
                        "nb_error": self._stats["nb_error"],
                    }

                self._stats["timestamps"].clear()
                self._stats["nb_error"] = 0
                stats_record_t0 = time.time()

            took = time.time() - start_t
            time.sleep(max(0, period - took))

    def set_torque(self, enabled: bool) -> None:
        assert self.c is not None, "Motor controller not initialized or already closed."

        if enabled:
            self.c.enable_torque()
        else:
            self.c.disable_torque()

        self._torque_enabled = enabled

    def close(self) -> None:
        self.c = None

    def get_stats(self) -> dict:
        if not hasattr(self, "_last_stats"):
            return {}

        return self._last_stats
