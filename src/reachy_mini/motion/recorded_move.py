import bisect  # noqa: D100
import json
import os
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
from huggingface_hub import snapshot_download

from reachy_mini.motion.move import Move
from reachy_mini.utils.interpolation import linear_pose_interpolation


def lerp(v0: float, v1: float, alpha: float) -> float:
    """Linear interpolation between two values."""
    return v0 + alpha * (v1 - v0)


class RecordedMove(Move):
    """Represent a recorded move."""

    def __init__(self, move: Dict[str, Any], sound_path: Optional[Path] = None) -> None:
        """Initialize RecordedMove."""
        self.move = move
        self._sound_path = sound_path

        self.description: str = self.move["description"]
        self.timestamps: List[float] = self.move["time"]
        self.trajectory: List[Dict[str, List[List[float]] | List[float] | float]] = (
            self.move["set_target_data"]
        )

        self.dt: float = (self.timestamps[-1] - self.timestamps[0]) / len(
            self.timestamps
        )

    @property
    def duration(self) -> float:
        """Get the duration of the recorded move."""
        return len(self.trajectory) * self.dt

    @property
    def sound_path(self) -> Optional[Path]:
        """Get the sound path associated with the move, if any."""
        return self._sound_path

    def evaluate(
        self, t: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
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

        head_prev = np.array(self.trajectory[idx_prev]["head"], dtype=np.float64)
        head_next = np.array(self.trajectory[idx_next]["head"], dtype=np.float64)
        antennas_prev: List[float] = self.trajectory[idx_prev]["antennas"]  # type: ignore[assignment]
        antennas_next: List[float] = self.trajectory[idx_next]["antennas"]  # type: ignore[assignment]
        body_yaw_prev: float = self.trajectory[idx_prev].get("body_yaw", 0.0)  # type: ignore[assignment]
        body_yaw_next: float = self.trajectory[idx_next].get("body_yaw", 0.0)  # type: ignore[assignment]
        # check_collision = self.trajectory[idx_prev].get("check_collision", False)

        # Interpolate to infer a better position at the current time.
        # Joint interpolations are easy:

        antennas_joints = np.array(
            [
                lerp(pos_prev, pos_next, alpha)
                for pos_prev, pos_next in zip(antennas_prev, antennas_next)
            ],
            dtype=np.float64,
        )

        body_yaw = lerp(body_yaw_prev, body_yaw_next, alpha)

        # Head position interpolation is more complex:
        head_pose = linear_pose_interpolation(head_prev, head_next, alpha)

        return head_pose, antennas_joints, body_yaw


class RecordedMoves:
    """Load a library of recorded moves from a HuggingFace dataset."""

    def __init__(self, hf_dataset_name: str):
        """Initialize RecordedMoves."""
        self.hf_dataset_name = hf_dataset_name
        self.local_path = snapshot_download(self.hf_dataset_name, repo_type="dataset")
        self.moves: Dict[str, Any] = {}
        self.sounds: Dict[str, Optional[Path]] = {}

        self.process()

    def process(self) -> None:
        """Populate recorded moves and sounds."""
        move_paths_tmp = glob(f"{self.local_path}/*.json")
        move_paths = [Path(move_path) for move_path in move_paths_tmp]
        for move_path in move_paths:
            move_name = move_path.stem

            move = json.load(open(move_path, "r"))
            self.moves[move_name] = move

            sound_path = move_path.with_suffix(".wav")
            self.sounds[move_name] = None

            if os.path.exists(sound_path):
                self.sounds[move_name] = sound_path

    def get(self, move_name: str) -> RecordedMove:
        """Get a recorded move by name."""
        if move_name not in self.moves:
            raise ValueError(
                f"Move {move_name} not found in recorded moves library {self.hf_dataset_name}"
            )

        return RecordedMove(self.moves[move_name], self.sounds[move_name])

    def list_moves(self) -> List[str]:
        """List all moves in the loaded library."""
        return list(self.moves.keys())
