import json
from pathlib import Path
from typing import Any, Dict, List

from reachy_mini.motion.recorded_move import RecordedMove

class LocalMoves:
    """Load a specific list of recorded moves from a local directory."""

    def __init__(self, local_path: str, move_list: List[str]):
        """
        Initialize LocalMoves.

        Args:
            local_path (str): Base path to the directory containing move JSON files.
            move_list (List[str]): The definitive list of move names to load.
        """
        self.local_path = Path(local_path)
        self.moves: Dict[str, Any] = {}
        self.process(move_list)

    def process(self, move_list: List[str]) -> None:
        """Load the specified moves from the local directory."""
        for move_name in move_list:
            move_path = self.local_path / f"{move_name}.json"
            if not move_path.exists():
                print(f"Warning: Move '{move_name}' listed in manifest but file not found at {move_path}")
                continue
            
            try:
                with open(move_path, "r") as f:
                    move = json.load(f)
                    self.moves[move_name] = move
            except Exception as e:
                print(f"Warning: Failed to load local move {move_name} from {move_path}: {e}")

    def get(self, move_name: str) -> RecordedMove:
        """Get a recorded move by name."""
        if move_name not in self.moves:
            raise ValueError(
                f"Move {move_name} not found in loaded library. Check manifest and file existence."
            )
        return RecordedMove(self.moves[move_name])

    def list_moves(self) -> List[str]:
        """List all successfully loaded moves."""
        return list(self.moves.keys())