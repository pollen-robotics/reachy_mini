from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import json


@dataclass
class ReachyMiniCommand:
    # Default height of the head in meters
    # which corresponds to the height when all joints are 0.
    DEFAULT_HEIGHT: float = 0.177

    head_pose: Optional[NDArray[np.float64]] = None  # shape: (4, 4)
    antennas_orientation: Optional[NDArray[np.float64]] = (
        None  # shape: (2,) [left, right]
    )

    def __init__(
        self,
        head_pose: Optional[NDArray[np.float64]] = None,
        antennas_orientation: Optional[NDArray[np.float64]] = None,
        offset_zero: bool = False,
    ) -> None:
        if head_pose is not None:
            if isinstance(head_pose, list):
                head_pose = np.array(head_pose, dtype=np.float64)
            if head_pose.shape != (4, 4):
                raise ValueError("head_pose must be a 4x4 matrix.")
        if antennas_orientation is not None:
            if isinstance(antennas_orientation, list):
                antennas_orientation = np.array(antennas_orientation, dtype=np.float64)
            if antennas_orientation.shape != (2,):
                raise ValueError("antennas_orientation must be a vector of size 2.")

        self.head_pose = head_pose.copy() if head_pose is not None else None

        if offset_zero and self.head_pose is not None:
            self.head_pose[2, 3] += self.DEFAULT_HEIGHT

        self.antennas_orientation = (
            antennas_orientation.copy() if antennas_orientation is not None else None
        )

    def update_with(self, other: "ReachyMiniCommand") -> None:
        """Update this command with another ReachyMiniCommand."""
        if other.head_pose is not None:
            self.head_pose = other.head_pose.copy()
        if other.antennas_orientation is not None:
            self.antennas_orientation = other.antennas_orientation.copy()

    def copy(self) -> "ReachyMiniCommand":
        """Return a copy of this command."""
        return ReachyMiniCommand(
            head_pose=self.head_pose.copy() if self.head_pose is not None else None,
            antennas_orientation=(
                self.antennas_orientation.copy()
                if self.antennas_orientation is not None
                else None
            ),
        )

    @classmethod
    def default(cls) -> "ReachyMiniCommand":
        """Return a default command with no pose and antennas."""
        default_head_pose = np.eye(4, dtype=np.float64)
        default_head_pose[2, 3] = cls.DEFAULT_HEIGHT

        return ReachyMiniCommand(
            head_pose=default_head_pose,
            antennas_orientation=np.zeros(2, dtype=np.float64),
        )

    @classmethod
    def from_json(cls, data: str) -> "ReachyMiniCommand":
        """Create a ReachyMiniCommand from a JSON string."""

        parsed = json.loads(data)

        head_pose = parsed.get("head_pose")
        if head_pose is not None:
            head_pose = np.array(head_pose, dtype=np.float64)

        antennas_orientation = parsed.get("antennas_orientation")
        if antennas_orientation is not None:
            antennas_orientation = np.array(antennas_orientation, dtype=np.float64)

        return ReachyMiniCommand(
            head_pose=head_pose,
            antennas_orientation=antennas_orientation,
        )

    def to_json(self) -> str:
        """Convert the command to a JSON string."""
        data = {
            "head_pose": (
                self.head_pose.tolist() if self.head_pose is not None else None
            ),
            "antennas_orientation": (
                self.antennas_orientation.tolist()
                if self.antennas_orientation is not None
                else None
            ),
        }
        return json.dumps(data)
