
import json
import time
from typing import List, Dict, Any

import numpy as np
import numpy.typing as npt

from reachy_mini.motion.move import Move
from reachy_mini.motion.recorded_move import RecordedMoves, RecordedMove

class Choreography(Move):
    """A composite move that sequences multiple RecordedMove objects based on a JSON definition."""

    def __init__(self, choreography_path: str, dances_library: RecordedMoves, emotions_library: RecordedMoves):
        """
        Initialize the Choreography move.

        Args:
            choreography_path (str): Path to the choreography JSON file.
            dances_library (RecordedMoves): An instance of RecordedMoves for dances.
            emotions_library (RecordedMoves): An instance of RecordedMoves for emotions.
        """
        with open(choreography_path, 'r') as f:
            choreography_data = json.load(f)

        self.bpm = choreography_data['bpm']
        self.sequence_data = choreography_data['sequence']
        self.dances_library = dances_library
        self.emotions_library = emotions_library

        # Load the definitive lists of moves
        with open('moves.json', 'r') as f:
            move_data = json.load(f)
            # Extract just the names from the new {name, description} structure
            dances_raw = move_data.get('dances', [])
            emotions_raw = move_data.get('emotions', [])
            # Handle both old (string array) and new (object array) formats
            self.move_lists = {
                'dances': [m['name'] if isinstance(m, dict) else m for m in dances_raw],
                'emotions': [m['name'] if isinstance(m, dict) else m for m in emotions_raw]
            }

        self.moves: List[RecordedMove] = []
        self.durations: List[float] = []
        self.start_times: List[float] = []

        # Use the final duration calculated by the LLM validator if available
        self._total_duration = choreography_data.get('final_duration', 0.0)
        if self._total_duration == 0.0:
            # Fallback to recalculating if not provided (e.g., old recommendations)
            self._prepare_sequence()
        else:
            # If final_duration is provided, we still need to prepare the sequence
            # but the _total_duration is already set.
            self._prepare_sequence_with_fixed_duration()

    def _prepare_sequence(self):
        """
        Load the moves from the library and calculate durations and start times.
        Calculate total duration from actual move durations.
        """
        current_time = 0.0

        for move_info in self.sequence_data:
            move_name = move_info.get('move') or move_info.get('move_name')

            # Skip manual moves (they're handled differently in the Choreography system)
            if move_name == 'manual' or move_name is None:
                continue

            cycles = move_info.get('cycles', 1)

            # Determine which library to use
            if move_name in self.move_lists['dances']:
                library = self.dances_library
            elif move_name in self.move_lists['emotions']:
                library = self.emotions_library
            else:
                raise ValueError(f"Move '{move_name}' not found in any known move list.")

            base_move = library.get(move_name)

            cycle_duration = base_move.duration

            for _ in range(cycles):
                self.moves.append(base_move)
                self.durations.append(cycle_duration)
                self.start_times.append(current_time)
                current_time += cycle_duration

        # Set total duration based on actual calculated duration
        self._total_duration = current_time

    def _prepare_sequence_with_fixed_duration(self):
        """
        Load the moves from the library and calculate durations and start times,
        but use the pre-calculated _total_duration.
        """
        current_time = 0.0
        for move_info in self.sequence_data:
            move_name = move_info.get('move') or move_info.get('move_name')
            if not move_name or move_name == 'manual' or move_name == 'idle':
                continue

            cycles = move_info.get('cycles', 1)
            
            # Determine which library to use
            if move_name in self.move_lists['dances']:
                library = self.dances_library
            elif move_name in self.move_lists['emotions']:
                library = self.emotions_library
            else:
                raise ValueError(f"Move '{move_name}' not found in any known move list.")

            base_move = library.get(move_name)
            
            cycle_duration = base_move.duration
            
            for _ in range(cycles):
                self.moves.append(base_move)
                self.durations.append(cycle_duration)
                self.start_times.append(current_time)
                current_time += cycle_duration

        # Ensure the last move's duration is adjusted if needed to match _total_duration
        if self.moves and current_time > self._total_duration:
            # Calculate the difference to trim
            diff_to_trim = current_time - self._total_duration
            
            # Shorten the last move's duration
            last_move_index = len(self.durations) - 1
            if self.durations[last_move_index] > diff_to_trim:
                self.durations[last_move_index] -= diff_to_trim
            else:
                # If the last move is too short to absorb the diff, remove it and adjust previous
                # This is a more complex scenario, for now, we'll just warn.
                print(f"[Choreography] WARNING: Last move too short to absorb duration difference. Total duration might still be off by {diff_to_trim:.2f}s")

        # If the choreography is too short, we don't add anything here, as the LLM should have filled it.
        # The _total_duration is already set, and the sequence is based on the LLM's output.

    def get_move_at_time(self, t: float) -> tuple[str, int, float, float]:
        """
        Returns the name, index, start time, and duration of the move active at time t.
        """
        if t < 0 or t > self._total_duration:
            return "", -1, 0.0, 0.0

        for i, start_time in enumerate(self.start_times):
            if start_time <= t < start_time + self.durations[i]:
                # Find corresponding sequence_data index
                # self.moves has expanded cycles, self.sequence_data has unique moves
                # Need to map back from expanded move index to original sequence
                seq_idx = 0
                cumulative_cycles = 0
                for seq_idx, seq_info in enumerate(self.sequence_data):
                    cycles = seq_info.get('cycles', 1)
                    if i < cumulative_cycles + cycles:
                        break
                    cumulative_cycles += cycles

                if seq_idx < len(self.sequence_data):
                    move_info = self.sequence_data[seq_idx]
                    move_name = move_info.get('move') or move_info.get('move_name', 'unknown')
                    return move_name, i, start_time, self.durations[i]

        # If past the last move, return info for the last move
        if self.moves:
            last_idx = len(self.moves) - 1
            # Find last sequence_data entry
            if self.sequence_data:
                move_info = self.sequence_data[-1]
                move_name = move_info.get('move') or move_info.get('move_name', 'unknown')
                return move_name, last_idx, self.start_times[last_idx], self.durations[last_idx]

        return "", -1, 0.0, 0.0

    @property
    def duration(self) -> float:
        """Return the total duration of the choreography."""
        return self._total_duration

    def evaluate(self, t: float) -> tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None, float | None]:
        """
        Evaluate the choreography at a specific time t.

        This method finds the active move at time t and evaluates it at its local time.
        """
        if t < 0 or t > self._total_duration:
            # Consider raising an error or handling this case more gracefully
            return None, None, None

        # Find which move is active at time t
        active_move_index = -1
        for i, start_time in enumerate(self.start_times):
            if start_time <= t < start_time + self.durations[i]:
                active_move_index = i
                break
        
        if active_move_index == -1:
            # This can happen at the very end of the choreography
            # Return the last pose of the last move
            last_move = self.moves[-1]
            return last_move.evaluate(last_move.duration)

        active_move = self.moves[active_move_index]
        move_start_time = self.start_times[active_move_index]
        
        # Calculate the local time for the active move
        local_time = t - move_start_time
        
        return active_move.evaluate(local_time)

