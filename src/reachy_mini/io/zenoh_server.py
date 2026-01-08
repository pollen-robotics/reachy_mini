"""Zenoh server for Reachy Mini.

This module implements a Zenoh server that allows communication with the Reachy Mini
robot. It handles commands for joint positions and torque settings, and publishes joint positions updates.

It uses the Zenoh protocol for efficient data exchange and can be configured to run
either on localhost only or to accept connections from other hosts.
"""

import asyncio
import json
import threading
from datetime import datetime

import numpy as np
import zenoh

from reachy_mini.daemon.backend.abstract import Backend, MotorControlMode
from reachy_mini.io.abstract import AbstractServer
from reachy_mini.io.protocol import (
    GotoTaskRequest,
    PlayMoveTaskRequest,
    TaskProgress,
    TaskRequest,
)


class ZenohServer(AbstractServer):
    """Zenoh server for Reachy Mini."""

    def __init__(self, prefix: str, backend: Backend, localhost_only: bool = True):
        """Initialize the Zenoh server."""
        self.prefix = prefix
        self.localhost_only = localhost_only
        self.backend = backend

        self._lock = threading.Lock()
        self._cmd_event = threading.Event()

    def start(self) -> None:
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
            c = zenoh.Config.from_json5(
                json.dumps(
                    {
                        # Listen on all interfaces → reachable on LAN/Wi-Fi
                        "listen": {
                            "endpoints": ["tcp/0.0.0.0:7447"],
                        },
                        # Allow standard discovery
                        "scouting": {
                            "multicast": {"enabled": True},
                            "gossip": {"enabled": True},
                        },
                        # No forced connect target; router will accept incoming sessions
                        "connect": {"endpoints": []},
                    }
                )
            )

        self.session = zenoh.open(c)
        self.sub = self.session.declare_subscriber(
            f"{self.prefix}/command",
            self._handle_command,
        )
        self.pub = self.session.declare_publisher(f"{self.prefix}/joint_positions")
        self.pub_record = self.session.declare_publisher(f"{self.prefix}/recorded_data")
        self.backend.set_joint_positions_publisher(self.pub)
        self.backend.set_recording_publisher(self.pub_record)

        self.pub_pose = self.session.declare_publisher(f"{self.prefix}/head_pose")
        self.backend.set_pose_publisher(self.pub_pose)

        # Declare IMU data publisher
        self.pub_imu = self.session.declare_publisher(f"{self.prefix}/imu_data")
        self.backend.set_imu_publisher(self.pub_imu)

        self.task_req_sub = self.session.declare_subscriber(
            f"{self.prefix}/task",
            self._handle_task_request,
        )
        self.task_progress_pub = self.session.declare_publisher(
            f"{self.prefix}/task_progress"
        )

        self.pub_status = self.session.declare_publisher(f"{self.prefix}/daemon_status")

    def stop(self) -> None:
        """Stop the Zenoh server."""
        self.session.close()  # type: ignore[no-untyped-call]

    def command_received_event(self) -> threading.Event:
        """Wait for a new command and return it."""
        return self._cmd_event

    def _handle_command(self, sample: zenoh.Sample) -> None:
        data = sample.payload.to_string()
        command = json.loads(data)
        with self._lock:
            block_targets = (
                self.backend.is_move_running
            )  # Prevent concurrent target updates from different clients

            def _maybe_ignore(field: str) -> bool:
                """Return True if the command should be ignored while a move runs."""
                if not block_targets:
                    return False
                self.backend.logger.warning(
                    f"Ignoring {field} command: a move is currently running."
                )
                return True

            if "torque" in command:
                if (
                    command["ids"] is not None
                ):  # If specific motor IDs are provided, just set torque for those motors
                    self.backend.set_motor_torque_ids(command["ids"], command["torque"])
                else:
                    if command["torque"]:
                        self.backend.set_motor_control_mode(MotorControlMode.Enabled)
                    else:
                        self.backend.set_motor_control_mode(MotorControlMode.Disabled)
            if "head_joint_positions" in command:
                if _maybe_ignore("head_joint_positions"):
                    pass
                else:
                    self.backend.set_target_head_joint_positions(
                        np.array(command["head_joint_positions"])
                    )
            if "head_pose" in command:
                if _maybe_ignore("head_pose"):
                    pass
                else:
                    self.backend.set_target_head_pose(
                        np.array(command["head_pose"]).reshape(4, 4)
                    )
            if "body_yaw" in command:
                if _maybe_ignore("body_yaw"):
                    pass
                else:
                    self.backend.set_target_body_yaw(command["body_yaw"])
            if "antennas_joint_positions" in command:
                if _maybe_ignore("antennas_joint_positions"):
                    pass
                else:
                    self.backend.set_target_antenna_joint_positions(
                        np.array(command["antennas_joint_positions"]),
                    )
            if "gravity_compensation" in command:
                try:
                    if command["gravity_compensation"]:
                        self.backend.set_motor_control_mode(
                            MotorControlMode.GravityCompensation
                        )
                    else:
                        self.backend.set_motor_control_mode(MotorControlMode.Enabled)

                except ValueError as e:
                    print(e)
            if "automatic_body_yaw" in command:
                self.backend.set_automatic_body_yaw(command["automatic_body_yaw"])

            if "set_target_record" in command:
                self.backend.append_record(command["set_target_record"])

            if "start_recording" in command:
                self.backend.start_recording()
            if "stop_recording" in command:
                self.backend.stop_recording()
        self._cmd_event.set()

    def _handle_task_request(self, sample: zenoh.Sample) -> None:
        task_req = TaskRequest.model_validate_json(sample.payload.to_string())

        if isinstance(task_req.req, GotoTaskRequest):
            req = task_req.req

            def task() -> None:
                asyncio.run(
                    self.backend.goto_target(
                        head=np.array(req.head).reshape(4, 4) if req.head else None,
                        antennas=np.array(req.antennas) if req.antennas else None,
                        duration=req.duration,
                        method=req.method,
                        body_yaw=req.body_yaw,
                    )
                )
        elif isinstance(task_req.req, PlayMoveTaskRequest):

            def task() -> None:
                print("PLAY MOVE")

        else:
            assert False, f"Unknown task request type {task_req.req.__class__.__name__}"

        def wrapped_task() -> None:
            error = None
            try:
                task()
            except Exception as e:
                error = str(e)

            progress = TaskProgress(
                uuid=task_req.uuid,
                finished=True,
                error=error,
                timestamp=datetime.now(),
            )
            self.task_progress_pub.put(progress.model_dump_json())

        threading.Thread(target=wrapped_task).start()
