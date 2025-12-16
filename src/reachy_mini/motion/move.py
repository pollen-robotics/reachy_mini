"""Module for defining motion moves on the ReachyMini robot."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt


class Move(ABC):
    """Abstract base class for defining a move on the ReachyMini robot."""

    @property
    def sound_path(self) -> Optional[Path]:
        """Get the sound path associated with the move, if any."""
        return None

    @property
    @abstractmethod
    def duration(self) -> float:
        """Duration of the move in seconds."""
        pass

    @abstractmethod
    def evaluate(
        self,
        t: float,
    ) -> tuple[
        npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None, float | None
    ]:
        """Evaluate the move at time t, typically called at a high-frequency (eg. 100Hz).

        Arguments:
            t: The time at which to evaluate the move (in seconds). It will always be between 0 and duration.

        Returns:
            head: The head position (4x4 homogeneous matrix).
            antennas: The antennas positions (rad).
            body_yaw: The body yaw angle (rad).

        """
        pass
