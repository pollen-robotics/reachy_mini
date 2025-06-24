import json
import time

from reachy_mini_motor_controller import ReachyMiniMotorController
from reachy_mini.io import Backend


class RobotBackend(Backend):
    def __init__(self, serialport: str):
        super().__init__()
        self.c = ReachyMiniMotorController(serialport)
        self.control_loop_frequency = 200.0
        self.publish_frequency = 100.0
        self.decimation = int(self.control_loop_frequency / self.publish_frequency)
        self.last_alive = time.time()

        self._torque_enabled = False

    def run(self):
        period = 1.0 / self.control_loop_frequency  # Control loop period in seconds
        step = 0

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
                    except RuntimeError as e:
                        print(f"Error reading positions: {e}")

                        if self.last_alive + 2 < time.time():
                            print("No response from the robot for 2 seconds, stopping.")
                            raise e

            took = time.time() - start_t
            time.sleep(max(0, period - took))

    def set_torque(self, enabled: bool) -> None:
        if enabled:
            self.c.enable_torque()
        else:
            self.c.disable_torque()

        self._torque_enabled = enabled
