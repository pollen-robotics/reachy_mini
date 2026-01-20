"""Common pydantic models definitions."""

from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy.spatial.transform import Rotation as R

from reachy_mini.daemon.backend.abstract import MotorControlMode


class Matrix4x4Pose(BaseModel):
    """Represent a 3D pose by its 4x4 transformation matrix (translation is expressed in meters)."""

    m: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]

    @classmethod
    def from_pose_array(cls, arr: NDArray[np.float64]) -> "Matrix4x4Pose":
        """Create a Matrix4x4 pose representation from a 4x4 pose array."""
        assert arr.shape == (4, 4), "Array must be of shape (4, 4)"
        m: tuple[
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
        ] = tuple(arr.flatten().tolist())
        return cls(m=m)

    def to_pose_array(self) -> NDArray[np.float64]:
        """Convert the Matrix4x4Pose to a 4x4 numpy array."""
        return np.array(self.m).reshape((4, 4))


class XYZRPYPose(BaseModel):
    """Represent a 3D pose using position (x, y, z) in meters and orientation (roll, pitch, yaw) angles in radians."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    @classmethod
    def from_pose_array(cls, arr: NDArray[np.float64]) -> "XYZRPYPose":
        """Create an XYZRPYPose representation from a 4x4 pose array."""
        assert arr.shape == (4, 4), "Array must be of shape (4, 4)"

        x, y, z = arr[0, 3], arr[1, 3], arr[2, 3]
        roll, pitch, yaw = R.from_matrix(arr[:3, :3]).as_euler("xyz")

        return cls(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )

    def to_pose_array(self) -> NDArray[np.float64]:
        """Convert the XYZRPYPose to a 4x4 numpy array."""
        rotation = R.from_euler("xyz", [self.roll, self.pitch, self.yaw])
        pose_matrix = np.eye(4)
        pose_matrix[:3, 3] = [self.x, self.y, self.z]
        pose_matrix[:3, :3] = rotation.as_matrix()
        return pose_matrix


AnyPose = XYZRPYPose | Matrix4x4Pose


def as_any_pose(pose: NDArray[np.float64], use_matrix: bool) -> AnyPose:
    """Convert a numpy array to an AnyPose representation."""
    return (
        Matrix4x4Pose.from_pose_array(pose)
        if use_matrix
        else XYZRPYPose.from_pose_array(pose)
    )


class FullBodyTarget(BaseModel):
    """Represent the full body including the head pose and the joints for antennas."""

    target_head_pose: AnyPose | None = None
    target_antennas: tuple[float, float] | None = None
    target_body_yaw: float | None = None
    timestamp: datetime | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "target_head_pose": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "roll": 0.0,
                        "pitch": 0.0,
                        "yaw": 0.0,
                    },
                    "target_antennas": [0.0, 0.0],
                    "target_body_yaw": 0.0,
                }
            ]
        }
    }


class DoAInfo(BaseModel):
    """Direction of Arrival info from the microphone array."""

    angle: float  # Angle in radians (0=left, π/2=front, π=right)
    speech_detected: bool


class FullState(BaseModel):
    """Represent the full state of the robot including all joint positions and poses."""

    control_mode: MotorControlMode | None = None
    head_pose: AnyPose | None = None
    head_joints: list[float] | None = None
    body_yaw: float | None = None
    antennas_position: list[float] | None = None
    timestamp: datetime | None = None
    passive_joints: list[float] | None = None
    doa: DoAInfo | None = None
