"""Base classes for server and client implementations.

These abstract classes define the interface for server and client components
in the Reachy Mini project. They provide methods for starting and stopping
the server, handling commands, and managing client connections.
"""

from abc import ABC, abstractmethod
from threading import Event
from typing import Any, Dict, List, Optional
from uuid import UUID

import numpy as np
import numpy.typing as npt

from reachy_mini.io.protocol import AnyCommand, AnyTaskRequest, DaemonStatus, ImuDataMsg


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
    def wait_for_connection(self, timeout: float = 5.0) -> None:
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
    def send_command(self, cmd: AnyCommand) -> None:
        """Send a typed command to the server."""
        pass

    @abstractmethod
    def get_current_joints(self) -> tuple[list[float], list[float]]:
        """Get the current joint positions."""
        pass

    @abstractmethod
    def get_current_head_pose(self) -> npt.NDArray[np.float64]:
        """Get the current head pose as a 4x4 matrix."""
        pass

    @abstractmethod
    def get_status(self, wait: bool = True, timeout: float = 5.0) -> DaemonStatus:
        """Get the last received daemon status."""
        pass

    @abstractmethod
    def get_current_imu_data(self) -> ImuDataMsg | None:
        """Get the current IMU data."""
        pass

    @abstractmethod
    def send_task_request(self, task_req: AnyTaskRequest) -> UUID:
        """Send a task request to the server and return a unique task identifier."""
        pass

    @abstractmethod
    def wait_for_task_completion(self, task_uid: UUID, timeout: float = 5.0) -> None:
        """Wait for the specified task to complete."""
        pass

    @abstractmethod
    def wait_for_recorded_data(self, timeout: float = 5.0) -> bool:
        """Block until the daemon publishes recorded data (or timeout)."""
        pass

    @abstractmethod
    def get_recorded_data(
        self, wait: bool = True, timeout: float = 5.0
    ) -> Optional[List[Dict[str, Any]]]:
        """Return the cached recording, optionally blocking until it arrives."""
        pass
