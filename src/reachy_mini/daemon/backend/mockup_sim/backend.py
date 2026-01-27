"""Mockup Simulation Backend for Reachy Mini.

A lightweight simulation backend that doesn't require MuJoCo.
Target positions become current positions immediately (no physics).
The kinematics engine is still used for FK/IK computations.

Apps open the webcam/microphone directly (like with a real robot).
"""

import json
import time
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import numpy.typing as npt

from ..abstract import Backend, MotorControlMode


class MockupSimBackend(Backend):
    """Lightweight simulated Reachy Mini without MuJoCo.

    This backend provides a simple simulation where target positions
    are applied immediately without physics simulation.

    Apps access webcam/microphone directly (not via UDP streaming).
    """

    def __init__(
        self,
        check_collision: bool = False,
        kinematics_engine: str = "AnalyticalKinematics",
        use_audio: bool = True,
    ) -> None:
        """Initialize the MockupSimBackend.

        Args:
            check_collision: If True, enable collision checking. Default is False.
            kinematics_engine: Kinematics engine to use. Defaults to "AnalyticalKinematics".
            use_audio: If True, use audio. Default is True.

        """
        super().__init__(
            check_collision=check_collision,
            kinematics_engine=kinematics_engine,
            use_audio=use_audio,
        )

        from reachy_mini.reachy_mini import (
            SLEEP_ANTENNAS_JOINT_POSITIONS,
            SLEEP_HEAD_JOINT_POSITIONS,
        )

        # Initialize with sleep positions
        self._head_joint_positions: npt.NDArray[np.float64] = np.array(
            SLEEP_HEAD_JOINT_POSITIONS, dtype=np.float64
        )
        self._antenna_joint_positions: npt.NDArray[np.float64] = np.array(
            SLEEP_ANTENNAS_JOINT_POSITIONS, dtype=np.float64
        )

        self._motor_control_mode = MotorControlMode.Enabled

        # Control loop frequency
        self.control_frequency = 50.0  # Hz

    def run(self) -> None:
        """Run the simulation loop.

        In mockup-sim mode, target positions are applied immediately.
        """
        control_period = 1.0 / self.control_frequency

        # Initialize kinematics with current positions
        self.update_head_kinematics_model(
            self._head_joint_positions,
            self._antenna_joint_positions,
        )

        while not self.should_stop.is_set():
            start_t = time.time()

            # Apply target positions immediately (no physics)
            if self.target_head_joint_positions is not None:
                self._head_joint_positions = self.target_head_joint_positions.copy()
            if self.target_antenna_joint_positions is not None:
                self._antenna_joint_positions = (
                    self.target_antenna_joint_positions.copy()
                )

            # Update current states
            self.current_head_joint_positions = self._head_joint_positions.copy()
            self.current_antenna_joint_positions = self._antenna_joint_positions.copy()

            # Update kinematics model (computes FK)
            self.update_head_kinematics_model(
                self.current_head_joint_positions,
                self.current_antenna_joint_positions,
            )

            # Update target head joint positions from IK if necessary
            if self.ik_required:
                try:
                    self.update_target_head_joints_from_ik(
                        self.target_head_pose, self.target_body_yaw
                    )
                except ValueError:
                    pass  # IK failed, keep current positions

            # Publish joint positions via Zenoh
            if (
                self.joint_positions_publisher is not None
                and self.pose_publisher is not None
                and not self.is_shutting_down
            ):
                self.joint_positions_publisher.put(
                    json.dumps(
                        {
                            "head_joint_positions": self.current_head_joint_positions.tolist(),
                            "antennas_joint_positions": self.current_antenna_joint_positions.tolist(),
                        }
                    ).encode("utf-8")
                )
                self.pose_publisher.put(
                    json.dumps(
                        {
                            "head_pose": self.get_present_head_pose().tolist(),
                        }
                    ).encode("utf-8")
                )

            self.ready.set()

            # Sleep to maintain control frequency
            elapsed = time.time() - start_t
            time.sleep(max(0, control_period - elapsed))

    def get_status(self) -> "MockupSimBackendStatus":
        """Get the status of the backend."""
        return MockupSimBackendStatus(motor_control_mode=self._motor_control_mode)

    def get_present_head_joint_positions(
        self,
    ) -> Annotated[npt.NDArray[np.float64], (7,)]:
        """Get the current joint positions of the head."""
        return self._head_joint_positions.copy()  # type: ignore[no-any-return]

    def get_present_antenna_joint_positions(
        self,
    ) -> Annotated[npt.NDArray[np.float64], (2,)]:
        """Get the current joint positions of the antennas."""
        return self._antenna_joint_positions.copy()  # type: ignore[no-any-return]

    def get_motor_control_mode(self) -> MotorControlMode:
        """Get the motor control mode."""
        return self._motor_control_mode

    def set_motor_control_mode(self, mode: MotorControlMode) -> None:
        """Set the motor control mode."""
        self._motor_control_mode = mode

    def set_motor_torque_ids(self, ids: list[str], on: bool) -> None:
        """Set the motor torque state for specific motor names.

        No-op in mockup-sim mode.
        """
        pass


@dataclass
class MockupSimBackendStatus:
    """Status of the MockupSim backend."""

    motor_control_mode: MotorControlMode
    error: str | None = None
