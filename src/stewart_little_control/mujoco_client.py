import socket
import pickle
import numpy as np
import time


class MujocoClient:
    def __init__(self, ip="127.0.0.1", port=1234):
        self.ip = ip
        self.port = port

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.ip, self.port))

    def send_pose(self, pose, offset_zero=True):
        """
        offset_zero : True if we consider that the resting position (motor zero) is the zero position of the robot.
                      False if the zero is the world zero
        """

        if offset_zero:
            pose[2] += 0.155

        data = pickle.dumps(pose)
        self.client_socket.sendall(data)


if __name__ == "__main__":
    client = MujocoClient()
    while True:
        pose = np.eye(4)
        pose[:3, 3][0] += 0.01 * np.sin(2 * np.pi * 0.5 * time.time())
        client.send_pose(pose)
        time.sleep(0.02)
