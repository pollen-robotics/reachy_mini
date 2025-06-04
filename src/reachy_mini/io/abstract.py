from abc import ABC, abstractmethod

from reachy_mini.command import ReachyMiniCommand


class AbstractServer(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        """Stop the server."""
        pass

    @abstractmethod
    def get_latest_command(self) -> ReachyMiniCommand:
        """Return the latest pose and antennas command."""
        pass


class AbstractClient(ABC):
    @abstractmethod
    def send_command(self, command: ReachyMiniCommand):
        """Send a command to the server.

        :param command: The command to send, containing head pose and antennas orientation.
        """
        pass
