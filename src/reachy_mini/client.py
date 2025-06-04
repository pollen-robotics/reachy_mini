import socket
import pickle


class Client:
    def __init__(self, ip="127.0.0.1", port=1234):
        self.ip = ip
        self.port = port

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.ip, self.port))

    def send_pose(self, _pose, antennas=None, offset_zero=False):
        """
        offset_zero : True if we consider that the resting position (motor zero) is the zero position of the robot.
                      False if the zero is the world zero
        """
        pose = _pose.copy()
        if offset_zero:
            pose[2, 3] += 0.177

        data = pickle.dumps({"pose": pose, "antennas": antennas})
        self.client_socket.sendall(data)
