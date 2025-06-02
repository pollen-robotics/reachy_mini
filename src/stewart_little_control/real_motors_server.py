from stewart_little_control import PlacoIK
import time
import os
from pathlib import Path
import numpy as np
import argparse

from reachy_mini_motor_controller import ReachyMiniMotorController

from stewart_little_control.io.abstract import AbstractServer
from stewart_little_control.io.socket_server import SocketServer


ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent


class RealMotorsServer:
    def __init__(self, serialport: str, server: AbstractServer):
        self.server = server

        self.placo_ik = PlacoIK(f"{ROOT_PATH}/descriptions/stewart_little_magnet/")
        self.current_pose = np.eye(4)
        self.current_pose[:3, 3][2] = 0.155
        self.current_antennas = np.zeros(2)

        # Start the hardware control loop
        self.run_motor_control_loop(serialport)

    def run_motor_control_loop(self, serialport: str, frequency: float = 100.0):
        c = ReachyMiniMotorController(serialport)
        c.enable_torque()
        period = 1.0 / frequency  # Control loop period in seconds

        try:
            while True:
                start_t = time.time()

                command = self.server.get_latest_command()

                try:
                    angles_rad = self.placo_ik.ik(command.head_pose)
                    # Removes antennas
                    angles_rad = angles_rad[:5] + angles_rad[-1:]
                    c.set_stewart_platform_position(angles_rad)
                except Exception as e:
                    print(f"IK error: {e}")

                c.set_antennas_positions(command.antennas_orientation)

                took = time.time() - start_t
                time.sleep(max(0, period - took))
        except KeyboardInterrupt:
            print("Stopping motor control loop.")
            c.disable_torque()


def main(args: argparse.Namespace):
    socket_server = SocketServer()
    socket_server.start()

    RealMotorsServer(args.serialport, socket_server)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the real motors server.")
    parser.add_argument(
        "serialport",
        type=str,
        help="The serial port to connect to the motors (e.g., /dev/ttyUSB0).",
    )
    args = parser.parse_args()

    main(args)
