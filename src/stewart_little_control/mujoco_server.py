import socket
import pickle
from stewart_little_control import PlacoIK
import mujoco
import mujoco.viewer
import time
import os
from pathlib import Path
from stewart_little_control.mujoco_utils import get_joint_qpos
import numpy as np
from threading import Thread, Lock

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent


class MujocoServer:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 1234
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

        self.model = mujoco.MjModel.from_xml_path(
            f"{ROOT_PATH}/descriptions/reachy_home_v1/scene.xml"
        )
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002  # s, simulation timestep, 500hz
        self.decimation = 10  # -> 50hz control loop

        self.placo_ik = PlacoIK(f"{ROOT_PATH}/descriptions/reachy_home_v1/")
        self.current_pose = np.eye(4)
        self.current_pose[:3, 3][2] = 0.177
        self.current_antennas = np.zeros(2)

        self.pose_lock = Lock()

        # Launch the client handler in a thread
        Thread(target=self.client_handler, daemon=True).start()

        # Start the simulation loop
        self.simulation_loop()

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

    def simulation_loop(self):
        step = 0
        all_start_t = time.time()
        with mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            while True:
                start_t = time.time()

                if step % self.decimation == 0:
                    with self.pose_lock:
                        pose = self.current_pose.copy()

                    # IK and apply control
                    try:
                        angles_rad = self.placo_ik.ik(pose)
                        self.data.ctrl[:] = angles_rad
                    except Exception as e:
                        print(f"IK error: {e}")

                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                took = time.time() - start_t
                time.sleep(max(0, self.model.opt.timestep - took))
                step += 1


def main():
    MujocoServer()


if __name__ == "__main__":
    main()
