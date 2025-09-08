import bisect  # noqa: D100
import json
from glob import glob
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from reachy_mini.utils.interpolation import linear_pose_interpolation

from . import Move


def lerp(v0, v1, alpha):
    """Linear interpolation between two values."""
    return v0 + alpha * (v1 - v0)


class RecordedMoves:
    """Load a library of recorded moves from a HuggingFace dataset."""

    def __init__(self, hf_dataset_name: str):
        """Initialize RecordedMoves."""
        self.hf_dataset_name = hf_dataset_name
        self.local_path = snapshot_download(self.hf_dataset_name, repo_type="dataset")
        self.moves = {}

        self.process()

    def process(self):
        """Populate recorded moves and sounds."""
        move_paths = glob(f"{self.local_path}/*.json")
        move_paths = [Path(move_path) for move_path in move_paths]
        for move_path in move_paths:
            move_name = move_path.stem

            move = json.load(open(move_path, "r"))
            self.moves[move_name] = move

    def get(self, move_name):
        """Get a recorded move by name."""
        if move_name not in self.moves:
            raise ValueError(
                f"Move {move_name} not found in recorded moves library {self.hf_dataset_name}"
            )

        return RecordedMove(self.moves[move_name])

    def list_moves(self):
        """List all moves in the loaded library."""
        return list(self.moves.keys())


class RecordedMove(Move):
    """Represent a recorded move."""

    def __init__(self, move):
        """Initialize RecordedMove."""
        self.move = move

        self.description = self.move["description"]
        self.timestamps = self.move["time"]
        self.trajectory = self.move["set_target_data"]

        self.dt = (self.timestamps[-1] - self.timestamps[0]) / len(self.timestamps)

    @property
    def duration(self) -> float:
        """Get the duration of the recorded move."""
        return len(self.trajectory) * self.dt

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Evaluate the move at time t.

        Returns:
            head: The head position (4x4 homogeneous matrix).
            antennas: The antennas positions (rad).
            body_yaw: The body yaw angle (rad).

        """
        # Under is Remi's emotions code, adapted
        if t >= self.timestamps[-1]:
            raise Exception("Tried to evaluate recorded move beyond its duration.")

        # Locate the right interval in the recorded time array.
        # 'index' is the insertion point which gives us the next timestamp.
        index = bisect.bisect_right(self.timestamps, t)
        # print(f"index: {index}, expected index: {t / self.dt:.0f}")
        idx_prev = index - 1 if index > 0 else 0
        idx_next = index if index < len(self.timestamps) else idx_prev

        t_prev = self.timestamps[idx_prev]
        t_next = self.timestamps[idx_next]

        # Avoid division by zero (if by any chance two timestamps are identical).
        if t_next == t_prev:
            alpha = 0.0
        else:
            alpha = (t - t_prev) / (t_next - t_prev)

        head_prev = np.array(self.trajectory[idx_prev]["head"])
        head_next = np.array(self.trajectory[idx_next]["head"])
        antennas_prev = self.trajectory[idx_prev]["antennas"]
        antennas_next = self.trajectory[idx_next]["antennas"]
        body_yaw_prev = self.trajectory[idx_prev].get("body_yaw", 0.0)
        body_yaw_next = self.trajectory[idx_next].get("body_yaw", 0.0)
        # check_collision = self.trajectory[idx_prev].get("check_collision", False)

        # Interpolate to infer a better position at the current time.
        # Joint interpolations are easy:
        antennas_joints = np.array(
            [
                lerp(pos_prev, pos_next, alpha)
                for pos_prev, pos_next in zip(antennas_prev, antennas_next)
            ]
        )
        body_yaw = lerp(body_yaw_prev, body_yaw_next, alpha)

        # Head position interpolation is more complex:
        head_pose = linear_pose_interpolation(head_prev, head_next, alpha)

        return head_pose, antennas_joints, body_yaw
