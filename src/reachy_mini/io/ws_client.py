"""WebSocket client for Reachy Mini.

Connects to the daemon's /ws/sdk endpoint and provides cached state,
fire-and-forget commands, and task request/progress tracking.
"""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import numpy as np
import numpy.typing as npt
import websockets.exceptions
import websockets.sync.client as ws_sync
from pydantic import ValidationError

from reachy_mini.io.abstract import AbstractClient
from reachy_mini.io.protocol import (
    AnyCommand,
    AnyTaskRequest,
    DaemonStatus,
    HeadPoseMsg,
    ImuDataMsg,
    JointPositionsMsg,
    RecordedDataMsg,
    TaskProgress,
    TaskRequest,
    server_msg_adapter,
)

logger = logging.getLogger(__name__)


class WSClient(AbstractClient):
    """WebSocket client for Reachy Mini."""

    def __init__(self, host: str = "localhost", port: int = 8000) -> None:
        """Initialize the WebSocket client.

        Args:
            host: Hostname or IP of the daemon.
            port: Port of the daemon's FastAPI server.

        """
        self.host = host
        self.port = port

        self.joint_position_received = threading.Event()
        self.head_pose_received = threading.Event()
        self.status_received = threading.Event()
        self.imu_data_received = threading.Event()

        self._last_joint_positions: JointPositionsMsg | None = None
        self._last_head_pose: HeadPoseMsg | None = None
        self._last_imu_data: ImuDataMsg | None = None
        self._last_recorded_data: RecordedDataMsg | None = None
        self._recorded_data_ready = threading.Event()
        self._is_alive = False
        self._last_status: DaemonStatus | None = None

        self.tasks: dict[UUID, TaskState] = {}
        self._tasks_lock = threading.Lock()

        self._ws: ws_sync.ClientConnection | None = None
        self._recv_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._heartbeat = threading.Event()

        uri = f"ws://{host}:{port}/ws/sdk"
        try:
            self._ws = ws_sync.connect(uri)
        except (OSError, websockets.exceptions.InvalidHandshake, TimeoutError) as e:
            raise ConnectionError(f"Failed to connect to {uri}: {e}") from e

        # Start receive loop in background thread
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def wait_for_connection(self, timeout: float = 5.0) -> None:
        """Wait for the client to receive initial data from the server.

        Args:
            timeout: Maximum time to wait for the connection in seconds.

        Raises:
            TimeoutError: If the connection is not established within the timeout.

        """
        start = time.time()
        while not self.joint_position_received.wait(
            timeout=1.0
        ) or not self.head_pose_received.wait(timeout=1.0):
            if time.time() - start > timeout:
                self.disconnect()
                raise TimeoutError(
                    "Timeout while waiting for connection with the server."
                )
            logger.info("Waiting for connection with the server...")

        self._is_alive = True
        self._check_alive_evt = threading.Event()
        threading.Thread(target=self._check_alive, daemon=True).start()

    def _check_alive(self) -> None:
        """Periodically check if the client is still connected."""
        while not self._stop_event.is_set():
            self._is_alive = self.is_connected()
            self._check_alive_evt.set()
            time.sleep(1.0)

    def is_connected(self) -> bool:
        """Check if the client is still receiving data from the server."""
        self._heartbeat.clear()
        return self._heartbeat.wait(timeout=1.0)

    def disconnect(self) -> None:
        """Disconnect the client from the server."""
        self._stop_event.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def send_command(self, cmd: AnyCommand) -> None:
        """Send a typed command to the server.

        Args:
            cmd: A validated command model (one of the AnyCommand types).

        Raises:
            ConnectionError: If the connection with the server is lost.

        """
        if not self._is_alive:
            raise ConnectionError("Lost connection with the server.")

        if self._ws is not None:
            self._ws.send(cmd.model_dump_json())

    # ------------------------------------------------------------------
    # Receive loop
    # ------------------------------------------------------------------

    def _recv_loop(self) -> None:
        """Background thread: read messages from the WebSocket."""
        assert self._ws is not None
        try:
            for raw in self._ws:
                if self._stop_event.is_set():
                    break
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                try:
                    msg = server_msg_adapter.validate_json(raw)
                except ValidationError:
                    continue
                self._heartbeat.set()
                self._dispatch(msg)
        except websockets.exceptions.ConnectionClosed:
            pass

    def _dispatch(self, msg: Any) -> None:
        """Route an incoming message to the appropriate handler."""
        if isinstance(msg, JointPositionsMsg):
            self._last_joint_positions = msg
            self.joint_position_received.set()
        elif isinstance(msg, HeadPoseMsg):
            self._last_head_pose = msg
            self.head_pose_received.set()
        elif isinstance(msg, ImuDataMsg):
            self._last_imu_data = msg
            self.imu_data_received.set()
        elif isinstance(msg, DaemonStatus):
            self._last_status = msg
            self.status_received.set()
        elif isinstance(msg, TaskProgress):
            with self._tasks_lock:
                task = self.tasks.get(msg.uuid)
            if task is not None:
                if msg.error:
                    task.error = msg.error
                if msg.finished:
                    task.event.set()
        elif isinstance(msg, RecordedDataMsg):
            self._last_recorded_data = msg
            self._recorded_data_ready.set()

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def get_current_joints(self) -> tuple[list[float], list[float]]:
        """Get the current joint positions."""
        assert self._last_joint_positions is not None, (
            "No joint positions received yet. Wait for the client to connect."
        )
        return (
            self._last_joint_positions.head_joint_positions.copy(),
            self._last_joint_positions.antennas_joint_positions.copy(),
        )

    def get_current_head_pose(self) -> npt.NDArray[np.float64]:
        """Get the current head pose as a 4x4 matrix."""
        assert self._last_head_pose is not None, "No head pose received yet."
        return np.array(self._last_head_pose.head_pose)

    def get_status(self, wait: bool = True, timeout: float = 5.0) -> DaemonStatus:
        """Get the last received daemon status."""
        if wait and not self.status_received.wait(timeout):
            raise TimeoutError("Status not received in time.")
        self.status_received.clear()
        assert self._last_status is not None
        return self._last_status

    def get_current_imu_data(self) -> ImuDataMsg | None:
        """Get the current IMU data.

        Returns:
            ImuDataMsg with accelerometer, gyroscope, quaternion, and temperature,
            or None if no data has been received yet or IMU is not available.

        """
        return self._last_imu_data

    def wait_for_recorded_data(self, timeout: float = 5.0) -> bool:
        """Block until the daemon publishes the frames (or timeout)."""
        return self._recorded_data_ready.wait(timeout)

    def get_recorded_data(
        self, wait: bool = True, timeout: float = 5.0
    ) -> Optional[List[Dict[str, Any]]]:
        """Return the cached recording, optionally blocking until it arrives.

        Raises `TimeoutError` if nothing shows up in time.
        """
        if wait and not self._recorded_data_ready.wait(timeout):
            raise TimeoutError("Recording not received in time.")
        self._recorded_data_ready.clear()
        if self._last_recorded_data is not None:
            return self._last_recorded_data.data.copy()
        return None

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def send_task_request(self, task_req: AnyTaskRequest) -> UUID:
        """Send a task request to the server."""
        if not self._is_alive:
            raise ConnectionError("Lost connection with the server.")

        task = TaskRequest(uuid=uuid4(), req=task_req, timestamp=datetime.now())
        with self._tasks_lock:
            self.tasks[task.uuid] = TaskState(event=threading.Event(), error=None)

        assert self._ws is not None
        self._ws.send(task.model_dump_json())

        return task.uuid

    def wait_for_task_completion(self, task_uid: UUID, timeout: float = 5.0) -> None:
        """Wait for the specified task to complete."""
        with self._tasks_lock:
            task = self.tasks.get(task_uid)
        if task is None:
            raise ValueError("Task not found.")

        task.event.wait(timeout)

        if not task.event.is_set():
            raise TimeoutError("Task did not complete in time.")
        if task.error is not None:
            with self._tasks_lock:
                del self.tasks[task_uid]
            raise Exception(f"Task failed with error: {task.error}")

        with self._tasks_lock:
            del self.tasks[task_uid]


@dataclass
class TaskState:
    """Represents the state of a task."""

    event: threading.Event
    error: str | None
