from reachy_mini.io import Backend
import os
from pathlib import Path
from reachy_mini_motor_controller import ReachyMiniMotorController
import time
import json
import numpy as np

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent


class RobotBackend(Backend):
    def __init__(self, serialport: str):
        super().__init__()
        self.c = ReachyMiniMotorController(serialport)
        self.control_loop_frequency = 300.0
        self.publish_frequency = 50.0
        self.decimation = int(self.control_loop_frequency / self.publish_frequency)

    def run(self):
        # self.c.enable_torque()
        # self.wake_up()
        period = 1.0 / self.control_loop_frequency  # Control loop period in seconds
        step = 0

        while True:
            start_t = time.time()
            if self.head_joint_positions is not None:
                self.c.set_stewart_platform_position(
                    # stewart platform angles are inverted in the real robot
                    list(-np.array(self.head_joint_positions[1:]))
                )
                self.c.set_body_rotation(self.head_joint_positions[0])
            if self.antenna_joint_positions is not None:
                self.c.set_antennas_positions(self.antenna_joint_positions)

            if step % self.decimation == 0:
                if self.joint_positions_publisher is not None:
                    self.joint_positions_publisher.put(
                        json.dumps(
                            {
                                "head_joint_positions": self.get_head_joint_positions(),
                                "antennas_joint_positions": self.get_antenna_joint_positions(),
                            }
                        )
                    )

            took = time.time() - start_t
            time.sleep(max(0, period - took))

    # TODO don't read two times all positions
    def get_head_joint_positions(self):
        positions = self.c.read_all_positions()
        yaw = positions[0]
        dofs = positions[3:]  # All other dofs
        return [yaw] + list(
            -np.array(dofs)
        )  # stewart platform angles are inverted in the real robot

    def get_antenna_joint_positions(self):
        positions = self.c.read_all_positions()
        antennas = positions[1:3]
        return list(antennas)

    def set_torque(self, enabled: bool) -> None:
        if enabled:
            self.c.enable_torque()
        else:
            self.c.disable_torque()
