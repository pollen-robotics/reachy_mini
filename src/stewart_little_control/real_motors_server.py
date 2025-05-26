import socket
import pickle
from stewart_little_control import PlacoIK
import time
import os
from pathlib import Path
import numpy as np
from threading import Thread, Lock
import argparse

from reachy_mini_motor_controller import ReachyMiniMotorController


ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent


class RealMotorsServer:
    def __init__(self, serialport: str):
        self.host = "0.0.0.0"
        self.port = 1234
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

        self.placo_ik = PlacoIK(f"{ROOT_PATH}/descriptions/stewart_little_magnet/")
        self.current_pose = np.eye(4)
        self.current_pose[:3, 3][2] = 0.155
        self.current_antennas = np.zeros(2)

        self.pose_lock = Lock()

        # Launch the client handler in a thread
        Thread(target=self.client_handler, daemon=True).start()

        # Start the hardware control loop
        self.run_motor_control_loop(serialport)

    def client_handler(self):
        while True:
            print("Waiting for connection on port", self.port)
            try:
                conn, address = self.server_socket.accept()
                print(f"Client connected from {address}")
                with conn:
                    while True:
                        try:
                            data = conn.recv(4096)
                            if not data:
                                print("Client disconnected")
                                break

                            pose_antennas = pickle.loads(data)
                            pose = pose_antennas["pose"]
                            antennas = pose_antennas["antennas"]
                            if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
                                with self.pose_lock:
                                    self.current_pose = pose
                                    if antennas is not None:
                                        self.current_antennas = antennas
                            else:
                                print("Received invalid pose data")

                        except (
                            ConnectionResetError,
                            EOFError,
                            pickle.PickleError,
                        ) as e:
                            print(f"Client error: {e}")
                            break

            except Exception as e:
                print(f"Server error: {e}")

    def run_motor_control_loop(self, serialport: str, frequency: float = 100.0):
        c = ReachyMiniMotorController(serialport)
        c.enable_torque()
        period = 1.0 / frequency  # Control loop period in seconds

        try:
            while True:
                start_t = time.time()

                with self.pose_lock:
                    pose = self.current_pose.copy()
                    antennas = self.current_antennas.copy()

                try:
                    angles_rad = self.placo_ik.ik(pose)
                    # Removes antennas
                    angles_rad = angles_rad[:5] + angles_rad[-1:]
                    c.set_stewart_platform_position(angles_rad)
                except Exception as e:
                    print(f"IK error: {e}")

                c.set_antennas_positions(antennas)
                # c.set_body_rotation

                took = time.time() - start_t
                time.sleep(max(0, period - took))
        except KeyboardInterrupt:
            print("Stopping motor control loop.")
            c.disable_torque()


def main(args: argparse.Namespace):
    RealMotorsServer(args.serialport)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the real motors server.")
    parser.add_argument(
        "serialport",
        type=str,
        help="The serial port to connect to the motors (e.g., /dev/ttyUSB0).",
    )
    args = parser.parse_args()

    main(args)
