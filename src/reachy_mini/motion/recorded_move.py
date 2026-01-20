import bisect  # noqa: D100
import json
import logging
import os
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from reachy_mini.motion.move import Move
from reachy_mini.utils.interpolation import linear_pose_interpolation

logger = logging.getLogger(__name__)

# Default datasets to preload at daemon startup
DEFAULT_DATASETS = [
    "pollen-robotics/reachy-mini-emotions-library",
    "pollen-robotics/reachy-mini-dances-library",
]


def preload_dataset(dataset_name: str) -> str | None:
    """Pre-download a HuggingFace dataset to local cache.

    This function downloads the dataset with network access, so it should be
    called during daemon startup (not during playback) to avoid blocking.

    Args:
        dataset_name: The HuggingFace dataset name (e.g., "pollen-robotics/reachy-mini-emotions-library")

    Returns:
        The local path to the cached dataset, or None if download failed.

    """
    try:
        logger.info(f"Pre-downloading dataset: {dataset_name}")
        local_path: str = snapshot_download(dataset_name, repo_type="dataset")
        logger.info(f"Dataset {dataset_name} cached at: {local_path}")
        return local_path
    except Exception as e:
        logger.warning(f"Failed to pre-download dataset {dataset_name}: {e}")
        return None


def preload_default_datasets() -> dict[str, str | None]:
    """Pre-download all default recorded move datasets.

    Should be called during daemon startup to ensure datasets are cached
    before any playback requests.

    Returns:
        A dict mapping dataset names to their local paths (or None if failed).

    """
    results = {}
    for dataset in DEFAULT_DATASETS:
        results[dataset] = preload_dataset(dataset)
    return results


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
    """Load a library of recorded moves from a HuggingFace dataset.

    Uses local cache only to avoid blocking network calls during playback.
    The dataset should be pre-downloaded at daemon startup via preload_default_datasets().
    If not cached, falls back to network download (which may cause delays).
    """

    def __init__(self, hf_dataset_name: str):
        """Initialize RecordedMoves."""
        self.hf_dataset_name = hf_dataset_name
        # Try local cache first (instant, no network)
        try:
            self.local_path = snapshot_download(
                self.hf_dataset_name,
                repo_type="dataset",
                local_files_only=True,
            )
        except LocalEntryNotFoundError:
            # Fallback: download from network (slow, but ensures it works)
            logger.warning(
                f"Dataset {hf_dataset_name} not in cache, downloading from HuggingFace. "
                "This may take a moment. Consider pre-loading datasets at daemon startup."
            )
            self.local_path = snapshot_download(
                self.hf_dataset_name,
                repo_type="dataset",
            )
        self.moves: Dict[str, Any] = {}
        self.sounds: Dict[str, Optional[Path]] = {}

        self.process()

    def process(self) -> None:
        """Populate recorded moves and sounds."""
        move_paths_tmp = glob(f"{self.local_path}/*.json")
        data_dir = os.path.join(self.local_path, "data")
        if os.path.isdir(data_dir):
            # Newer datasets keep their moves inside data/; look there as well.
            move_paths_tmp.extend(glob(f"{data_dir}/*.json"))
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
