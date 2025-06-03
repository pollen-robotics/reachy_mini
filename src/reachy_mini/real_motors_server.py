import socket
import pickle
from reachy_mini import PlacoKinematics
import time
import os
from pathlib import Path
import numpy as np
from threading import Thread, Lock
import argparse
import simpleaudio as sa

from reachy_mini_motor_controller import ReachyMiniMotorController
from reachy_mini.utils import minimum_jerk
from scipy.spatial.transform import Rotation as R


ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent

wave_obj = sa.WaveObject.from_wave_file(f"{ROOT_PATH}/src/assets/proud2.wav")


def play_sound():
    wave_obj.play()  # Non-blocking


class RealMotorsServer:
    def __init__(self, serialport: str):
        self.host = "0.0.0.0"
        self.port = 1234
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

        self.placo_kinematics = PlacoKinematics(
            f"{ROOT_PATH}/descriptions/reachy_mini/urdf/"
        )
        self.current_pose = np.eye(4)
        self.current_pose[:3, 3][2] = 0.177
        self.current_antennas = np.zeros(2)

        self.pose_lock = Lock()

        self.sleep_positions = [
            0.0,
            -0.849,
            1.292,
            -0.472,
            -0.047,
            -1.31,
            0.876,
            -3.05,
            3.05,
        ]

        self.sleep_positions[-2:] = [3.05, -3.05]  # Set antennas to sleep position
        self.init_pose = np.eye(4)
        self.init_pose[:3, 3][2] = 0.177  # Set the height of the head

        self.c = ReachyMiniMotorController(serialport)

        # Launch the client handler in a thread
        Thread(target=self.client_handler, daemon=True).start()

        # Start the hardware control loop
        self.run_motor_control_loop(frequency=300.0)

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

    def get_current_positions(self):
        positions = self.c.read_all_positions()
        yaw = positions[0]
        antennas = positions[1:3]
        dofs = positions[3:]  # All other dofs
        return [yaw] + list(dofs) + list(antennas)

    def goto_joints(self, present_position_rad, target_position_rad, duration=4):
        interp = minimum_jerk(
            np.array(present_position_rad.copy()),
            np.array(target_position_rad.copy()),
            duration,
        )
        t0 = time.time()
        while time.time() - t0 < duration:
            t = time.time() - t0
            angles_rad = interp(t)

            yaw_rad = angles_rad[0]
            stewart_rad = angles_rad[1:-2]
            antennas_rad = angles_rad[-2:]

            self.c.set_body_rotation(yaw_rad)
            self.c.set_stewart_platform_position(stewart_rad)
            self.c.set_antennas_positions(antennas_rad)
            time.sleep(0.01)

    def goto_sleep(self):
        current_positions_rad = self.get_current_positions()
        init_positions_rad = self.placo_kinematics.ik(self.init_pose.copy())
        try:
            self.goto_joints(current_positions_rad, init_positions_rad, duration=2)
        except KeyboardInterrupt:
            self.c.disable_torque()

        time.sleep(1.0)

        current_positions_rad = self.get_current_positions()
        try:
            self.goto_joints(current_positions_rad, self.sleep_positions, duration=2)
        except KeyboardInterrupt:
            self.c.disable_torque()

        self.c.disable_torque()

    def wake_up(self):
        current_positions_rad = self.get_current_positions()
        init_positions_rad = self.placo_kinematics.ik(self.init_pose.copy())
        init_positions_rad[-2:] = [0, 0]  # Set antennas to sleep position
        try:
            self.goto_joints(
                current_positions_rad.copy(), init_positions_rad.copy(), duration=2
            )
        except KeyboardInterrupt:
            self.c.disable_torque()

        print(self.placo_kinematics.fk(self.get_current_positions()))

        time.sleep(0.2)

        target_pose = np.eye(4)
        target_pose[:3, 3][2] = 0.177  # Set the height of the head
        euler_rot = [np.deg2rad(20), 0, 0]
        rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
        target_pose[:3, :3] = rot_mat
        target_positions_rad = self.placo_kinematics.ik(target_pose)

        play_sound()
        current_positions_rad = self.get_current_positions()
        try:
            self.goto_joints(current_positions_rad, target_positions_rad, duration=0.2)
        except KeyboardInterrupt:
            self.c.disable_torque()

        current_positions_rad = self.get_current_positions()
        try:
            self.goto_joints(current_positions_rad, init_positions_rad, duration=0.2)
        except KeyboardInterrupt:
            self.c.disable_torque()

    def run_motor_control_loop(self, frequency: float = 100.0):
        print(self.current_pose)
        self.c.enable_torque()
        self.wake_up()
        print(self.current_pose)
        period = 1.0 / frequency  # Control loop period in seconds

        try:
            while True:
                start_t = time.time()

                with self.pose_lock:
                    pose = self.current_pose.copy()
                    antennas = self.current_antennas.copy()
                try:
                    angles_rad = self.placo_kinematics.ik(pose)
                    stewart_angles_rad = angles_rad[1:-2]  # Exclude yaw and antennas
                    yaw_angles_rad = angles_rad[0]  # First angle is yaw
                    self.c.set_stewart_platform_position(stewart_angles_rad)
                    self.c.set_body_rotation(yaw_angles_rad)
                except Exception as e:
                    print(f"IK error: {e}")

                self.c.set_antennas_positions(antennas)

                took = time.time() - start_t
                time.sleep(max(0, period - took))
        except KeyboardInterrupt:
            print("Stopping motor control loop.")
            self.goto_sleep()
            # self.c.disable_torque()


def main(args: argparse.Namespace = "/dev/ttyACM0"):
    RealMotorsServer(args.serialport)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the real motors server.")
    parser.add_argument(
        "-s",
        "--serialport",
        type=str,
        help="The serial port to connect to the motors (e.g., /dev/ttyUSB0).",
        default="/dev/ttyACM0",
    )
    args = parser.parse_args()

    main(args)
