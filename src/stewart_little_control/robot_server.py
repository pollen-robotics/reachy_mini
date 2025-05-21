import socket
import pickle
import time
import numpy as np
from threading import Thread, Lock
from stewart_little_control import StewartLittleControl


class RobotServer:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 1234
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

        self.current_pose = np.eye(4)
        self.current_pose[:3, 3][2] = 0.155

        self.control_freq = 50  # Hz
        self.control = StewartLittleControl()

        self.pose_lock = Lock()

        # Launch the client handler in a thread
        Thread(target=self.client_handler, daemon=True).start()

        # Start the simulation loop
        self.control_loop()

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
                            # anennas = pose_antennas["antennas"]
                            if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
                                with self.pose_lock:
                                    self.current_pose = pose
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

    def control_loop(self):
        while True:
            start_t = time.time()

            with self.pose_lock:
                pose = self.current_pose.copy()

            # IK and apply control
            try:
                self.control.set_pose(pose)
            except Exception as e:
                print(f"IK error: {e}")

            took = time.time() - start_t
            time.sleep(max(0, (1.0 / self.control_freq) - took))


def main():
    RobotServer()


if __name__ == "__main__":
    main()
