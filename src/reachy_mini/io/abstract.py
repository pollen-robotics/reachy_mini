from abc import ABC, abstractmethod
from threading import Event


class AbstractServer(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        """Stop the server."""
        pass

    @abstractmethod
    def command_received_event(self) -> Event:
        """Wait for a new command and return it."""
        pass


class AbstractClient(ABC):
    @abstractmethod
    def send_command(self, command: str):
        """Send a command to the server."""
        pass
