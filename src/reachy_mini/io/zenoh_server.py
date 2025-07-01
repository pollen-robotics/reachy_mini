import json
import threading

import zenoh

from reachy_mini.io.abstract import AbstractServer
from reachy_mini.io.backend import Backend


class ZenohServer(AbstractServer):
    def __init__(self, backend: Backend, localhost_only: bool = True):  # type: ignore
        self.localhost_only = localhost_only
        self.backend = backend

        self._lock = threading.Lock()
        self._cmd_event = threading.Event()

    def start(self):
        if self.localhost_only:
            c = zenoh.Config.from_json5(
                json.dumps(
                    {
                        "listen": {
                            "endpoints": ["tcp/localhost:7447"],
                        },
                        "scouting": {
                            "multicast": {
                                "enabled": False,
                            },
                            "gossip": {
                                "enabled": False,
                            },
                        },
                        "connect": {
                            "endpoints": [
                                "tcp/localhost:7447",
                            ],
                        },
                    }
                )
            )
        else:
            c = zenoh.Config()

        self.session = zenoh.open(c)
        self.sub = self.session.declare_subscriber(
            "reachy_mini/command",
            self._handle_command,
        )
        self.pub = self.session.declare_publisher("reachy_mini/joint_positions")
        self.backend.set_joint_positions_publisher(self.pub)

    def stop(self):
        self.session.close()

    def command_received_event(self) -> threading.Event:
        """Wait for a new command and return it."""
        return self._cmd_event

    def _handle_command(self, sample: zenoh.Sample):
        data = sample.payload.to_string()
        command = json.loads(data)
        with self._lock:
            if "torque" in command:
                self.backend.set_torque(command["torque"])
            if "head_joint_positions" in command:
                self.backend.set_head_joint_positions(command["head_joint_positions"])
            if "antennas_joint_positions" in command:
                self.backend.set_antenna_joint_positions(
                    command["antennas_joint_positions"]
                )
        self._cmd_event.set()
