import socket
import pickle
import numpy as np
import time
from scipy.spatial.transform import Rotation as R


class Client:
    def __init__(self, ip="127.0.0.1", port=1234):
        self.ip = ip
        self.port = port

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.ip, self.port))

    def send_pose(self, _pose, antennas=None, offset_zero=False):
        pose = _pose.copy()
        if offset_zero:
            pose[2, 3] += 0.155

        message = {
            "type": "pose",
            "data": {
                "pose": pose,
                "antennas": antennas
            }
        }
        self.client_socket.sendall(pickle.dumps(message))
    
    def get_joint_positions(self):
        message = {"type": "get_joints"}
        self.client_socket.sendall(pickle.dumps(message))
        response = self.client_socket.recv(4096)
        data = pickle.loads(response)
        if data["type"] == "joints":
            # print(data)
            return data["data"]
        else:
            print("Unexpected response from server.")
            return None
        
    def send_joints(self, joints):
        """
        Send joint positions to the server.

        Args:
            joints (np.ndarray or list): Joint positions to send.
        """
        message = {
            "type": "joints",
            "data": np.array(joints)
        }
        self.client_socket.sendall(pickle.dumps(message))



if __name__ == "__main__":
    client = Client()
    while True:
        pose = np.eye(4)
        # euler_rot = [
        #     0,
        #     0,
        #     # 0.2 * np.sin(2 * np.pi * 0.5 * t),
        #     1.0 * np.sin(2 * np.pi * 0.5 * time.time() + np.pi),
        # ]
        # rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
        # pose[:3, :3] = rot_mat
        pose[:3, 3][2] += 0.01 * np.sin(2 * np.pi * 0.5 * time.time())
        antennas = np.array([1, 1]) * np.sin(2 * np.pi * 1.5 * time.time())
        client.send_pose(pose, antennas=antennas,offset_zero=True)
        time.sleep(0.02)
