import zenoh

from stewart_little_control.command import ReachyMiniCommand
from stewart_little_control.io.abstract import AbstractClient


class ZenohClient(AbstractClient):
    def __init__(self):
        self.session = zenoh.open(zenoh.Config())
        self.cmd_pub = self.session.declare_publisher("stewart_little/command")

    def send_command(self, command: ReachyMiniCommand):
        self.cmd_pub.put(command.to_json().encode("utf-8"))
