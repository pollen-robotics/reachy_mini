import time
from abc import ABC, abstractmethod
from multiprocessing import Event

import numpy as np

from reachy_mini.reachy_mini import ReachyMini


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


class MovePlayer:
    def play(
        self,
        reachy_mini: ReachyMini,
        move: Move,
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
