import json
import zenoh

from reachy_mini.command import ReachyMiniCommand
from reachy_mini.io.abstract import AbstractClient


class ZenohClient(AbstractClient):
    def __init__(self, scouting: bool = False):
        c = zenoh.Config.from_json5(
            json.dumps(
                {"scouting": {"multicast": {"enabled": scouting}}},
            )
        )

        self.session = zenoh.open(c)
        self.cmd_pub = self.session.declare_publisher("reachy_mini/command")

    def send_command(self, command: ReachyMiniCommand):
        self.cmd_pub.put(command.to_json().encode("utf-8"))
