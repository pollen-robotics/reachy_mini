"""Define the DanceMove and Choreography classes for ReachyMini.

These classes allow you to play dance moves and choreographies on the ReachyMini robot.
"""

import json
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from reachy_mini import ReachyMini
    from reachy_mini.daemon.backend.abstract import Backend


from reachy_mini.motion import Move
from reachy_mini.motion.dance.collection.dance import AVAILABLE_MOVES
from reachy_mini.utils import create_head_pose


class DanceMove(Move):
    """A specific type of Move that represents a dance move."""

    default_bpm: int = 114

    def __init__(self, move_name: str, **params):
        """Initialize a dance move with the given name and parameters."""
        self.move_fn, self.move_params, self.move_metadata = AVAILABLE_MOVES[move_name]
        self.move_params.update(params)

        self.name = move_name

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

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Evaluate the dance move at time t.

        This method calls the move function with the current time and parameters,
        returning the head and antennas positions.

        Args:
            t: The current time in seconds.

        Returns:
            A tuple containing the head position, antennas positions, body_yaw and the path to a sound to be played (if any, else None).

        """
        t_beats = t * (self.default_bpm / 60.0)  # Convert time to beats
        offsets = self.move_fn(t_beats, **self.move_params)
        (x, y, z) = offsets.position_offset
        (roll, pitch, yaw) = offsets.orientation_offset

        head_pose = create_head_pose(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, degrees=False, mm=False
        )
        return head_pose, offsets.antennas_offset, 0


class Choreography(Move):
    """A sequence of DanceMoves that form a choreography."""

    def __init__(self, choreography_file: str):
        """Initialize a choreography from a json file."""
        with open(choreography_file, "r") as f:
            choreography = json.load(f)

        self.bpm = choreography.get("bpm", DanceMove.default_bpm)

        self.moves = []
        self.cycles = []

        for move in choreography["sequence"]:
            move_name = move["move"]

            move_params = move.copy()
            move_params.pop("move", None)

            self.cycles.append(move_params.pop("cycles", 1))
            self.moves.append(DanceMove(move_name, **move_params))

    @property
    def duration(self) -> float:
        """Calculate the total duration of the choreography."""
        return sum(move.duration for move in self.moves)

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Evaluate the choreography at time t.

        This method iterates through the moves and evaluates them based on the
        current time, returning the head and antennas positions.

        Args:
            t: The current time in seconds.

        Returns:
            A tuple containing the head position and antennas positions.

        """
        raise NotImplementedError("Choreography evaluation is not implemented yet.")

    def play_on(
        self,
        reachy_mini: "Backend | ReachyMini",
        repeat: int = 1,
        frequency: float = 100,
        start_goto: bool = False,
        is_relative: bool = False,
    ):
        """Play the choreography on the ReachyMini robot."""
        for _ in range(repeat):
            for move, cycle in zip(self.moves, self.cycles):
                move.play_on(reachy_mini, repeat=cycle, frequency=frequency)
