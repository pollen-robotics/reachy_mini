import argparse
import os
import time
from pathlib import Path
import asyncio

import mujoco
import mujoco.viewer

from reachy_mini import PlacoKinematics, UDPJPEGFrameSender
from reachy_mini.io import Server

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
        self.rendering_timestep = 0.04  # s, rendering loop # 25Hz

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

        self.streamer_udp = UDPJPEGFrameSender()

        self.stop_event = asyncio.Event()

    def stop(self):
        self.stop_event.set()

    async def rendering_loop(self):
        while not self.stop_event.is_set():
            start_t = time.time()
            self.offscreen_renderer.update_scene(self.data, self.camera_id)
            im = self.offscreen_renderer.render()
            self.streamer_udp.send_frame(im)

            took = time.time() - start_t
            await asyncio.sleep(max(0, self.rendering_timestep - took))


    async def simulation_loop(self):
        step = 0
        all_start_t = time.time()
        with mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            while not self.stop_event.is_set():
                start_t = time.time()

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
                await asyncio.sleep(max(0, self.model.opt.timestep - took))
                step += 1

async def async_loop():
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

    mujoco_server = MujocoServer(scene=args.scene, server=server)

    try:
        await asyncio.gather(
                mujoco_server.rendering_loop(),
                mujoco_server.simulation_loop()
            )
    except KeyboardInterrupt:
        pass
    except asyncio.CancelledError:
        pass
    finally:
        mujoco_server.stop()

    server.stop()

def main():
    asyncio.run(async_loop())

if __name__ == "__main__":
    main()
