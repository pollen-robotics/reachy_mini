"""Base classes for server and client implementations.

These abstract classes define the interface for server and client components
in the Reachy Mini project. They provide methods for starting and stopping
the server, handling commands, and managing client connections.
"""

from abc import ABC, abstractmethod
from threading import Event
from uuid import UUID

from reachy_mini.io.protocol import AnyTaskRequest


class AbstractServer(ABC):
    """Base class for server implementations."""

    @abstractmethod
    def start(self) -> None:
        """Start the server."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the server."""
        pass

    @abstractmethod
    def command_received_event(self) -> Event:
        """Wait for a new command and return it."""
        pass


class AbstractClient(ABC):
    """Base class for client implementations."""

    @abstractmethod
    def wait_for_connection(self) -> None:
        """Wait for the client to connect to the server."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the client is connected to the server."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect the client from the server."""
        pass

    @abstractmethod
    def send_command(self, command: str) -> None:
        """Send a command to the server."""
        pass

    @abstractmethod
    def get_current_joints(self) -> tuple[list[float], list[float]]:
        """Get the current joint positions."""
        pass

    @abstractmethod
    def send_task_request(self, task_req: AnyTaskRequest) -> UUID:
        """Send a task request to the server and return a unique task identifier."""
        pass

    @abstractmethod
    def wait_for_task_completion(self, task_uid: UUID, timeout: float = 5.0) -> None:
        """Wait for the specified task to complete."""
        pass
