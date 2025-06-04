import threading
import zenoh

from reachy_mini.command import ReachyMiniCommand
from reachy_mini.io.abstract import AbstractServer


class ZenohServer(AbstractServer):
    def __init__(self):
        self._lock = threading.Lock()
        self._last_command = ReachyMiniCommand.default()

    def start(self):
        self.session = zenoh.open(zenoh.Config())
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

    def _handle_command(self, sample: zenoh.Sample):
        data = sample.payload.to_string()
        new_cmd = ReachyMiniCommand.from_json(data)
        with self._lock:
            self._last_command.update_with(new_cmd)
