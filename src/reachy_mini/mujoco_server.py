import argparse
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from reachy_mini import PlacoKinematics
from reachy_mini import UDPJPEGFrameSender
from reachy_mini.io import Server

import numpy as np

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent


class MujocoServer:
    def __init__(self, server: Server, scene="empty"):
        self.server = server

        self.model = mujoco.MjModel.from_xml_path(
            f"{ROOT_PATH}/descriptions/reachy_mini/mjcf/scenes/{scene}.xml"
        )
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002  # s, simulation timestep, 500hz
        self.decimation = 10  # -> 50hz control loop

        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "eye_camera"
        )
        self.camera_size = (1280, 720)
        self.offscreen_renderer = mujoco.Renderer(
            self.model, height=self.camera_size[1], width=self.camera_size[0]
        )

        self.placo_kinematics = PlacoKinematics(
            f"{ROOT_PATH}/descriptions/reachy_mini/urdf/", sim=True
        )

        # Start the simulation loop
        self.simulation_loop()

    def get_camera(self):
        self.offscreen_renderer.update_scene(self.data, self.camera_id)
        im = self.offscreen_renderer.render()
        return im

    def simulation_loop(self):
        step = 0
        all_start_t = time.time()
        with mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            while True:
                start_t = time.time()

                im = self.get_camera()
                self.streamer_udp.send_frame(im)

                if step % self.decimation == 0:
                    command = self.server.get_latest_command()

                    # IK and apply control
                    try:
                        angles_rad = self.placo_kinematics.ik(command.head_pose)
                        self.data.ctrl[:7] = angles_rad[:-2]
                        self.data.ctrl[-2:] = command.antennas_orientation
                    except Exception as e:
                        print(f"IK error: {e}")

                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                took = time.time() - start_t
                time.sleep(max(0, self.model.opt.timestep - took))
                step += 1


def main():
    parser = argparse.ArgumentParser(
        description="Launch the MuJoCo server with an optional scene specification."
    )
    parser.add_argument(
        "--scene",
        "-s",
        type=str,
        default="empty",
        help="Name of the scene to load (default: empty)",
    )
    args = parser.parse_args()

    server = Server()
    server.start()

    try:
        MujocoServer(scene=args.scene, server=server)
    except KeyboardInterrupt:
        pass

    server.stop()


if __name__ == "__main__":
    main()
