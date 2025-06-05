import json
import threading
import zenoh

from reachy_mini.command import ReachyMiniCommand
from reachy_mini.io.abstract import AbstractServer


class ZenohServer(AbstractServer):
    def __init__(self, localhost_only: bool = True):
        self.localhost_only = localhost_only

        self._lock = threading.Lock()
        self._last_command = ReachyMiniCommand.default()
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

    def stop(self):
        self.session.close()

    def get_latest_command(self) -> ReachyMiniCommand:
        """Return the latest pose and antennas command."""
        with self._lock:
            return self._last_command

    def command_received_event(self) -> threading.Event:
        """Wait for a new command and return it."""
        return self._cmd_event

    def _handle_command(self, sample: zenoh.Sample):
        data = sample.payload.to_string()
        new_cmd = ReachyMiniCommand.from_json(data)
        with self._lock:
            self._last_command.update_with(new_cmd)
        self._cmd_event.set()
