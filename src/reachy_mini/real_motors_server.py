import argparse
import os
import time
from pathlib import Path

import numpy as np
import pygame
from reachy_mini_motor_controller import ReachyMiniMotorController
from scipy.spatial.transform import Rotation as R

from reachy_mini import PlacoKinematics
from reachy_mini.command import ReachyMiniCommand
from reachy_mini.io import Server
from reachy_mini.utils import minimum_jerk

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent

pygame.mixer.init()
pygame.mixer.music.load(f"{ROOT_PATH}/src/assets/proud2.wav")


def play_sound():
    pygame.mixer.music.play()


class RealMotorsServer:
    def __init__(self, serialport: str, server: Server):
        self.server = server

        self.placo_kinematics = PlacoKinematics(
            f"{ROOT_PATH}/descriptions/reachy_mini/urdf/"
        )

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
        self.init_pose = ReachyMiniCommand.default().head_pose

        self.c = ReachyMiniMotorController(serialport)

        # Start the hardware control loop
        self.run_motor_control_loop(frequency=300.0)

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

        # If current position is far from initial position, move to initial position first
        if (
            np.linalg.norm(
                np.array(current_positions_rad) - np.array(init_positions_rad)
            )
            > 0.2
        ):
            try:
                self.goto_joints(current_positions_rad, init_positions_rad, duration=1)
            except KeyboardInterrupt:
                self.c.disable_torque()

            time.sleep(0.2)

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

        time.sleep(0.1)

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
        self.c.enable_torque()
        self.wake_up()
        period = 1.0 / frequency  # Control loop period in seconds

        try:
            while True:
                start_t = time.time()

                command = self.server.get_latest_command()

                try:
                    angles_rad = self.placo_kinematics.ik(command.head_pose)
                    stewart_angles_rad = angles_rad[1:-2]  # Exclude yaw and antennas
                    yaw_angles_rad = angles_rad[0]  # First angle is yaw
                    self.c.set_stewart_platform_position(stewart_angles_rad)
                    self.c.set_body_rotation(yaw_angles_rad)
                except Exception as e:
                    print(f"IK error: {e}")

                self.c.set_antennas_positions(command.antennas_orientation)

                took = time.time() - start_t
                time.sleep(max(0, period - took))
        except KeyboardInterrupt:
            print("Stopping motor control loop.")
            self.goto_sleep()
            self.server.stop()
            # self.c.disable_torque()


def main():
    parser = argparse.ArgumentParser(description="Run the real motors server.")
    parser.add_argument(
        "-s",
        "--serialport",
        type=str,
        help="The serial port to connect to the motors (e.g., /dev/ttyUSB0).",
        default="/dev/ttyACM0",
    )
    args = parser.parse_args()

    server = Server()
    server.start()

    RealMotorsServer(args.serialport, server)


if __name__ == "__main__":
    main()
