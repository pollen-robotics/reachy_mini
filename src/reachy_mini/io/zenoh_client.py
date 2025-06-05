import json
import zenoh

from reachy_mini.command import ReachyMiniCommand
from reachy_mini.io.abstract import AbstractClient


class ZenohClient(AbstractClient):
    def __init__(self, localhost_only: bool = True):
        if localhost_only:
            c = zenoh.Config.from_json5(
                json.dumps(
                    {
                        "connect": {
                            "endpoints": {
                                "peer": ["tcp/localhost:7447"],
                                "router": [],
                            },
                        },
                    }
                )
            )
        else:
            c = zenoh.Config()

        self.session = zenoh.open(c)
        self.cmd_pub = self.session.declare_publisher("reachy_mini/command")

    def send_command(self, command: ReachyMiniCommand):
        self.cmd_pub.put(command.to_json().encode("utf-8"))
