"""Zenoh server for Reachy Mini.

This module implements a Zenoh server that allows communication with the Reachy Mini
robot. It handles commands for joint positions and torque settings, and publishes joint positions updates.

It uses the Zenoh protocol for efficient data exchange and can be configured to run
either on localhost only or to accept connections from other hosts.
"""

import json
import threading

import numpy as np
import zenoh

from reachy_mini.daemon.backend.abstract import Backend
from reachy_mini.io.abstract import AbstractServer


class ZenohServer(AbstractServer):
    """Zenoh server for Reachy Mini."""

    def __init__(self, backend: Backend, localhost_only: bool = True):  # type: ignore
        """Initialize the Zenoh server."""
        self.localhost_only = localhost_only
        self.backend = backend

        self._lock = threading.Lock()
        self._cmd_event = threading.Event()

    def start(self):
        """Start the Zenoh server."""
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
        self.pub = self.session.declare_publisher("reachy_mini/joint_positions")
        self.backend.set_joint_positions_publisher(self.pub)

        self.pub_pose = self.session.declare_publisher("reachy_mini/head_pose")
        self.backend.set_pose_publisher(self.pub_pose)

    def stop(self):
        """Stop the Zenoh server."""
        self.session.close()

    def command_received_event(self) -> threading.Event:
        """Wait for a new command and return it."""
        return self._cmd_event

    def _handle_command(self, sample: zenoh.Sample):
        data = sample.payload.to_string()
        command = json.loads(data)
        with self._lock:
            if "torque" in command:
                if command["torque"]:
                    self.backend.enable_motors()
                else:
                    self.backend.disable_motors()
            if "head_joint_positions" in command:
                self.backend.set_target_head_joint_positions(
                    command["head_joint_positions"]
                )
            if "head_pose" in command:
                self.backend.set_target_head_pose(
                    np.array(command["head_pose"]).reshape(4, 4), command["body_yaw"]
                )
            if "antennas_joint_positions" in command:
                self.backend.set_target_antenna_joint_positions(
                    command["antennas_joint_positions"]
                )
            if "head_joint_current" in command:
                self.backend.set_target_head_joint_current(
                    command["head_joint_current"]
                )
            if "head_operation_mode" in command:
                self.backend.set_head_operation_mode(command["head_operation_mode"])
            if "antennas_operation_mode" in command:
                self.backend.set_antennas_operation_mode(
                    command["antennas_operation_mode"]
                )
            if "check_collision" in command:
                self.backend.set_check_collision(command["check_collision"])
            if "gravity_compensation" in command:
                self.backend.set_gravity_compensation_mode(
                    command["gravity_compensation"]
                )
            if "automatic_body_yaw" in command:
                self.backend.set_automatic_body_yaw(command["automatic_body_yaw"])

        self._cmd_event.set()
