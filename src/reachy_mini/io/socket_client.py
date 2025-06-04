import socket

from stewart_little_control.command import ReachyMiniCommand
from stewart_little_control.io.abstract import AbstractClient


class SocketClient(AbstractClient):
    def __init__(self, ip="127.0.0.1", port=1234):
        self.ip = ip
        self.port = port

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.ip, self.port))

    def send_command(self, command: ReachyMiniCommand):
        self.client_socket.sendall(command.to_json().encode("utf-8"))
