"""Zenoh client for Reachy Mini.

This module implements a Zenoh client that allows communication with the Reachy Mini
robot. It subscribes to joint positions updates and allows sending commands to the robot.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import numpy as np
import numpy.typing as npt
import zenoh

from reachy_mini.io.abstract import AbstractClient
from reachy_mini.io.protocol import AnyTaskRequest, TaskProgress, TaskRequest


class ZenohClient(AbstractClient):
    """Zenoh client for Reachy Mini."""

    def __init__(self, prefix: str, localhost_only: bool = True):
        """Initialize the Zenoh client.

        Args:
            prefix: The Zenoh prefix to use for communication (used to identify multiple robots).
            localhost_only: If True, connect to localhost only

        """
        self.prefix = prefix

        if localhost_only:
            c = zenoh.Config.from_json5(
                json.dumps(
                    {"mode": "client", "connect": {"endpoints": ["tcp/localhost:7447"]}}
                )
            )
        else:
            # Use peer mode with automatic discovery via multicast/gossip scouting
            # This allows the client to discover robots on the network without knowing their IP/hostname
            # The prefix/robot_name is used for topic namespacing only
            c = zenoh.Config.from_json5(
                json.dumps(
                    {
                        "mode": "peer",
                        "scouting": {
                            "multicast": {"enabled": True},
                            "gossip": {"enabled": True},
                        },
                        "connect": {"endpoints": []},
                    }
                )
            )

        self.joint_position_received = threading.Event()
        self.head_pose_received = threading.Event()
        self.status_received = threading.Event()
        self.imu_data_received = threading.Event()

        self.session = zenoh.open(c)
        self.cmd_pub = self.session.declare_publisher(f"{self.prefix}/command")

        self.joint_sub = self.session.declare_subscriber(
            f"{self.prefix}/joint_positions",
            self._handle_joint_positions,
        )

        self.pose_sub = self.session.declare_subscriber(
            f"{self.prefix}/head_pose",
            self._handle_head_pose,
        )

        self.recording_sub = self.session.declare_subscriber(
            f"{self.prefix}/recorded_data",
            self._handle_recorded_data,
        )

        self.status_sub = self.session.declare_subscriber(
            f"{self.prefix}/daemon_status",
            self._handle_status,
        )

        self.imu_sub = self.session.declare_subscriber(
            f"{self.prefix}/imu_data",
            self._handle_imu_data,
        )

        self._last_head_joint_positions = None
        self._last_antennas_joint_positions = None
        self._last_head_pose: Optional[npt.NDArray[np.float64]] = None
        self._recorded_data: Optional[
            List[Dict[str, float | List[float] | List[List[float]]]]
        ] = None
        self._recorded_data_ready = threading.Event()
        self._is_alive = False
        self._last_status: Dict[str, Any] = {}  # contains a DaemonStatus
        self._last_imu_data: Optional[Dict[str, List[float] | float]] = None

        self.tasks: dict[UUID, TaskState] = {}
        self.task_request_pub = self.session.declare_publisher(f"{self.prefix}/task")
        self.task_progress_sub = self.session.declare_subscriber(
            f"{self.prefix}/task_progress",
            self._handle_task_progress,
        )

    def wait_for_connection(self, timeout: float = 5.0) -> None:
        """Wait for the client to connect to the server.

        Args:
            timeout (float): Maximum time to wait for the connection in seconds.

        Raises:
            TimeoutError: If the connection is not established within the timeout period.

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
            logging.info("Waiting for connection with the server...")

        self._is_alive = True
        self._check_alive_evt = threading.Event()
        threading.Thread(target=self.check_alive, daemon=True).start()

    def check_alive(self) -> None:
        """Periodically check if the client is still connected to the server."""
        while True:
            self._is_alive = self.is_connected()
            self._check_alive_evt.set()
            time.sleep(1.0)

    def is_connected(self) -> bool:
        """Check if the client is connected to the server."""
        self.joint_position_received.clear()
        self.head_pose_received.clear()
        return self.joint_position_received.wait(
            timeout=1.0
        ) and self.head_pose_received.wait(timeout=1.0)

    def disconnect(self) -> None:
        """Disconnect the client from the server."""
        self.session.close()  # type: ignore[no-untyped-call]

    def send_command(self, command: str) -> None:
        """Send a command to the server."""
        if not self._is_alive:
            raise ConnectionError("Lost connection with the server.")

        self.cmd_pub.put(command.encode("utf-8"))

    def _handle_joint_positions(self, sample: zenoh.Sample) -> None:
        """Handle incoming joint positions."""
        if sample.payload:
            positions = json.loads(sample.payload.to_string())
            self._last_head_joint_positions = positions.get("head_joint_positions")
            self._last_antennas_joint_positions = positions.get(
                "antennas_joint_positions"
            )
            self.joint_position_received.set()

    def _handle_recorded_data(self, sample: zenoh.Sample) -> None:
        """Handle incoming recorded data."""
        print("Received recorded data.")
        if sample.payload:
            data = json.loads(sample.payload.to_string())
            self._recorded_data = data
            self._recorded_data_ready.set()
        if self._recorded_data is not None:
            print(f"Recorded data: {len(self._recorded_data)} frames received.")

    def _handle_status(self, sample: zenoh.Sample) -> None:
        """Handle incoming status updates."""
        if sample.payload:
            status = json.loads(sample.payload.to_string())
            self._last_status = status
            self.status_received.set()

    def _handle_imu_data(self, sample: zenoh.Sample) -> None:
        """Handle incoming IMU data."""
        if sample.payload:
            imu_data = json.loads(sample.payload.to_string())
            self._last_imu_data = imu_data
            self.imu_data_received.set()

    def get_current_joints(self) -> tuple[list[float], list[float]]:
        """Get the current joint positions."""
        assert (
            self._last_head_joint_positions is not None
            and self._last_antennas_joint_positions is not None
        ), "No joint positions received yet. Wait for the client to connect."
        return (
            self._last_head_joint_positions.copy(),
            self._last_antennas_joint_positions.copy(),
        )

    def wait_for_recorded_data(self, timeout: float = 5.0) -> bool:
        """Block until the daemon publishes the frames (or timeout)."""
        return self._recorded_data_ready.wait(timeout)

    def get_recorded_data(
        self, wait: bool = True, timeout: float = 5.0
    ) -> Optional[List[Dict[str, float | List[float] | List[List[float]]]]]:
        """Return the cached recording, optionally blocking until it arrives.

        Raises `TimeoutError` if nothing shows up in time.
        """
        if wait and not self._recorded_data_ready.wait(timeout):
            raise TimeoutError("Recording not received in time.")
        self._recorded_data_ready.clear()  # ready for next run
        if self._recorded_data is not None:
            return self._recorded_data.copy()
        return None

    def get_status(self, wait: bool = True, timeout: float = 5.0) -> Dict[str, Any]:
        """Get the last received status. Returns DaemonStatus as a dict."""
        if wait and not self.status_received.wait(timeout):
            raise TimeoutError("Status not received in time.")
        self.status_received.clear()  # ready for next run
        return self._last_status

    def get_current_imu_data(self) -> Optional[Dict[str, List[float] | float]]:
        """Get the current IMU data.

        Returns:
            dict with 'accelerometer', 'gyroscope', 'quaternion', and 'temperature' keys,
            or None if no data has been received yet or IMU is not available.

        """
        if self._last_imu_data is None:
            return None
        return self._last_imu_data.copy()

    def _handle_head_pose(self, sample: zenoh.Sample) -> None:
        """Handle incoming head pose."""
        if sample.payload:
            pose = json.loads(sample.payload.to_string())
            self._last_head_pose = np.array(pose.get("head_pose")).reshape(4, 4)
            self.head_pose_received.set()

    def get_current_head_pose(self) -> npt.NDArray[np.float64]:
        """Get the current head pose."""
        assert self._last_head_pose is not None, "No head pose received yet."
        return self._last_head_pose.copy()  # type: ignore[no-any-return]

    def send_task_request(self, task_req: AnyTaskRequest) -> UUID:
        """Send a task request to the server."""
        if not self._is_alive:
            raise ConnectionError("Lost connection with the server.")

        task = TaskRequest(uuid=uuid4(), req=task_req, timestamp=datetime.now())

        self.tasks[task.uuid] = TaskState(event=threading.Event(), error=None)

        self.task_request_pub.put(task.model_dump_json())

        return task.uuid

    def wait_for_task_completion(self, task_uid: UUID, timeout: float = 5.0) -> None:
        """Wait for the specified task to complete."""
        if task_uid not in self.tasks:
            raise ValueError("Task not found.")

        self.tasks[task_uid].event.wait(timeout)

        if not self.tasks[task_uid].event.is_set():
            raise TimeoutError("Task did not complete in time.")
        if self.tasks[task_uid].error is not None:
            raise Exception(f"Task failed with error: {self.tasks[task_uid].error}")

        del self.tasks[task_uid]

    def _handle_task_progress(self, sample: zenoh.Sample) -> None:
        if sample.payload:
            progress = TaskProgress.model_validate_json(sample.payload.to_string())
            assert progress.uuid in self.tasks, "Unknown task UUID."

            if progress.error:
                self.tasks[progress.uuid].error = progress.error

            if progress.finished:
                self.tasks[progress.uuid].event.set()


@dataclass
class TaskState:
    """Represents the state of a task."""

    event: threading.Event
    error: str | None
