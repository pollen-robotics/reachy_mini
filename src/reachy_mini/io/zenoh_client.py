"""Zenoh client for Reachy Mini.

This module implements a Zenoh client that allows communication with the Reachy Mini
robot. It subscribes to joint positions updates and allows sending commands to the robot.
"""

import json
import threading
import time

import numpy as np
import zenoh

from reachy_mini.io.abstract import AbstractClient


class ZenohClient(AbstractClient):
    """Zenoh client for Reachy Mini."""

    def __init__(self, localhost_only: bool = True):
        """Initialize the Zenoh client."""
        if localhost_only:
            c = zenoh.Config.from_json5(
                json.dumps(
                    {
                        "connect": {
                            "endpoints": {
                                "peer": ["tcp/localhost:7447"],
                                "router": [],
                            },
                        },
                    }
                )
            )
        else:
            c = zenoh.Config()

        self.session = zenoh.open(c)
        self.cmd_pub = self.session.declare_publisher("reachy_mini/command")

        self.joint_sub = self.session.declare_subscriber(
            "reachy_mini/joint_positions",
            self._handle_joint_positions,
        )
        self.pose_sub = self.session.declare_subscriber(
            "reachy_mini/head_pose",
            self._handle_head_pose,
        )
        self._last_head_joint_positions = None
        self._last_antennas_joint_positions = None
        self._last_head_pose = None
        self.keep_alive_event = threading.Event()

    def wait_for_connection(self, timeout: float = 5.0):
        """Wait for the client to connect to the server.

        Args:
            timeout (float): Maximum time to wait for the connection in seconds.

        Raises:
            TimeoutError: If the connection is not established within the timeout period.

        """
        start = time.time()
        while not self.keep_alive_event.wait(timeout=1.0):
            if time.time() - start > timeout:
                self.disconnect()
                raise TimeoutError(
                    "Timeout while waiting for connection with the server."
                )
            print("Waiting for connection with the server...")

    def is_connected(self) -> bool:
        """Check if the client is connected to the server."""
        self.keep_alive_event.clear()
        return self.keep_alive_event.wait(timeout=1.0)

    def disconnect(self):
        """Disconnect the client from the server."""
        self.session.close()

    def send_command(self, command: str):
        """Send a command to the server."""
        self.cmd_pub.put(command.encode("utf-8"))

    def _handle_joint_positions(self, sample):
        """Handle incoming joint positions."""
        if sample.payload:
            positions = json.loads(sample.payload.to_string())
            self._last_head_joint_positions = positions.get("head_joint_positions")
            self._last_antennas_joint_positions = positions.get(
                "antennas_joint_positions"
            )
            self.keep_alive_event.set()

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

    def _handle_head_pose(self, sample):
        """Handle incoming head pose."""
        if sample.payload:
            pose = json.loads(sample.payload.to_string())
            self._last_head_pose = np.array(pose.get("head_pose")).reshape(4, 4)
            self.keep_alive_event.set()

    def get_current_head_pose(self) -> np.ndarray:
        """Get the current head pose."""
        assert self._last_head_pose is not None, "No head pose received yet."
        return self._last_head_pose.copy()
