from reachy_mini.io import Backend
import os
from pathlib import Path
from reachy_mini import PlacoKinematics
from reachy_mini_motor_controller import ReachyMiniMotorController
import time
import json

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent


class RobotBackend(Backend):
    def __init__(self, serialport: str):
        super().__init__()
        self.placo_kinematics = PlacoKinematics(
            f"{ROOT_PATH}/descriptions/reachy_mini/urdf/"
        )
        self.c = ReachyMiniMotorController(serialport)
