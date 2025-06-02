from stewart_little_control import PlacoIK
import mujoco
import mujoco.viewer
import time
import os
from pathlib import Path
from stewart_little_control.io.abstract import AbstractServer


ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent


class MujocoServer:
    def __init__(self, server: AbstractServer):
        self.server = server

        self.model = mujoco.MjModel.from_xml_path(
            f"{ROOT_PATH}/descriptions/stewart_little_magnet/scene.xml"
        )
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002  # s, simulation timestep, 500hz
        self.decimation = 10  # -> 50hz control loop

        self.placo_ik = PlacoIK(f"{ROOT_PATH}/descriptions/stewart_little_magnet/")

        # Start the simulation loop
        self.simulation_loop()

    def simulation_loop(self):
        step = 0
        all_start_t = time.time()
        with mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            while True:
                start_t = time.time()

                if step % self.decimation == 0:
                    command = self.server.get_latest_command()

                    # IK and apply control
                    try:
                        angles_rad = self.placo_ik.ik(command.head_pose)
                        self.data.ctrl[:] = angles_rad
                        self.data.ctrl[5:7] = command.antennas_orientation
                    except Exception as e:
                        print(f"IK error: {e}")

                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                took = time.time() - start_t
                time.sleep(max(0, self.model.opt.timestep - took))
                step += 1


def main():
    from stewart_little_control.io import Server

    server = Server()
    server.start()

    try:
        MujocoServer(server)
    except KeyboardInterrupt:
        pass

    server.stop()


if __name__ == "__main__":
    main()
