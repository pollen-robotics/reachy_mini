import socket
import pickle
from stewart_little_control import PlacoIK
import mujoco
import mujoco.viewer
import time
import os
from pathlib import Path
from stewart_little_control.mujoco_utils import get_joint_qpos, get_joints
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
            f"{ROOT_PATH}/descriptions/stewart_little_magnet/scene.xml"
        )
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002  # s, simulation timestep, 500hz
        self.decimation = 10  # -> 50hz control loop

        self.placo_ik = PlacoIK(f"{ROOT_PATH}/descriptions/stewart_little_magnet/")
        self.current_pose = np.eye(4)
        self.current_pose[:3, 3][2] = 0.155
        self.current_antennas = np.zeros(2)
        self.control_mode = "pose"  # default
        self.last_received_joints = np.zeros_like(self.data.ctrl)


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
                Thread(target=self.handle_client, args=(conn,), daemon=True).start()
            except Exception as e:
                print(f"Server error: {e}")


    def handle_client(self, conn):
        with conn:
            while True:
                try:
                    data = conn.recv(4096)
                    if not data:
                        print("Client disconnected")
                        break

                    message = pickle.loads(data)

                    if message["type"] == "pose":
                        pose = message["data"]["pose"]
                        antennas = message["data"].get("antennas", None)
                        if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
                            with self.pose_lock:
                                self.current_pose = pose
                                if antennas is not None:
                                    self.current_antennas = antennas
                                self.control_mode = "pose"
                        else:
                            print("Invalid pose format")

                    elif message["type"] == "joints":
                        joints = message["data"]
                        if isinstance(joints, (list, np.ndarray)):
                            with self.pose_lock:
                                self.last_received_joints = np.array(joints)
                                self.control_mode = "joints"
                    elif message["type"] == "get_joints":
                        # Send back joint positions
                        print("Sending joint positions")
                        qpos = get_joints(self.model, self.data)
                        response = pickle.dumps({"type": "joints", "data": qpos})
                        conn.sendall(response)

                except (ConnectionResetError, EOFError, pickle.PickleError) as e:
                    print(f"Client error: {e}")
                    break

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
                        antennas = self.current_antennas.copy()

                    # IK and apply control
                    try:
                        if self.control_mode == "pose":
                            angles_rad = self.placo_ik.ik(pose)
                            self.data.ctrl[:] = angles_rad
                            self.data.ctrl[5:7] = antennas
                        elif self.control_mode == "joints":
                            self.data.ctrl[:] = self.last_received_joints
                    except Exception as e:
                        print(f"Control error: {e}")

                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                took = time.time() - start_t
                time.sleep(max(0, self.model.opt.timestep - took))
                step += 1


def main():
    MujocoServer()


if __name__ == "__main__":
    main()
