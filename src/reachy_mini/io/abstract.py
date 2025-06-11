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
    def wait_for_connection(self):
        """Wait for the client to connect to the server."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the client is connected to the server."""
        pass

    @abstractmethod
    def send_command(self, command: str):
        """Send a command to the server."""
        pass

    @abstractmethod
    def get_current_joints(self) -> tuple[list[float], list[float]]:
        """Get the current joint positions."""
        pass
