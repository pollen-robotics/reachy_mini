import time
from abc import ABC, abstractmethod
from multiprocessing import Event

import numpy as np

from reachy_mini.reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.moves import AVAILABLE_MOVES


class Move(ABC):
    @property
    @abstractmethod
    def duration(self) -> float:
        """Duration of the move in seconds."""
        pass

    @abstractmethod
    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the move at time t.

        Returns:
            head: The head position.
            antennas: The antennas positions.

        """


class DanceMove(Move):
    """A specific type of Move that represents a dance move."""

    default_bpm: int = 114

    def __init__(self, move_name: str, **params):
        self.move_fn, self.move_params, self.move_metadata = AVAILABLE_MOVES[move_name]
        self.move_params.update(params)

    @property
    def duration(self) -> float:
        """Return the duration of the dance move.

        The duration is calculated based on the default BPM (beats per minute).
        Each move is assumed to be one beat long, and the duration is computed
        as the time taken for one beat in seconds.
        """
        f = self.default_bpm / 60.0  # Convert BPM to Hz
        beat_duration = 1.0 / f  # Duration of one beat in seconds
        nb_beats = self.move_metadata.get("default_duration_beats", 1)

        return beat_duration * nb_beats

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the dance move at time t.

        This method calls the move function with the current time and parameters,
        returning the head and antennas positions.

        Args:
            t: The current time in seconds.

        Returns:
            A tuple containing the head position and antennas positions.

        """
        offsets = self.move_fn(t, **self.move_params)
        (x, y, z) = offsets.position_offset
        (roll, pitch, yaw) = offsets.orientation_offset

        head_pose = create_head_pose(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, degrees=False, mm=False
        )
        return head_pose, offsets.antennas_offset


class MovePlayer:
    def play(
        self,
        reachy_mini: ReachyMini,
        move: Move,
        repeat: int = 1,
        frequency: float = 100.0,
    ):
        timer = Event()
        dt = 1.0 / frequency

        t0 = time.time()

        while True:
            t = time.time() - t0

            if t > move.duration:
                break

            head, antennas = move.evaluate(t)
            reachy_mini.set_target(head=head, antennas=antennas)

            loop_duration = time.time() - t
            sleep_duration = max(0, dt - loop_duration)
            if sleep_duration > 0:
                timer.wait(sleep_duration)


if __name__ == "__main__":
    import numpy as np

    from reachy_mini import ReachyMini
    from reachy_mini.utils.move_player import DanceMove, MovePlayer

    possible_moves = list(AVAILABLE_MOVES.keys())

    with ReachyMini() as reachy:
        while True:
            move = DanceMove(np.random.choice(possible_moves))
            print(move.move_metadata["description"])

            MovePlayer().play(reachy, move)
