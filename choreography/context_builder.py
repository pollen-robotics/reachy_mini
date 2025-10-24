"""
Choreography Context Builder
Loads move metadata including beat counts for BPM-based choreography generation
Similar to parameter_prep pattern from main LAURA system
"""

import json
from typing import Dict, List, Any
from reachy_mini.motion.recorded_move import RecordedMoves, RecordedMove
from choreography.move_metadata import get_beat_count, calculate_move_duration


class ChoreographyContext:
    """Builds context with actual move metadata for LLM choreography generation."""

    # Dataset names
    DANCES_DATASET = "pollen-robotics/reachy-mini-dances-library"
    EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"

    def __init__(self):
        """Initialize context builder and load datasets."""
        print("[ChoreographyContext] Loading move datasets...")
        self.dances = RecordedMoves(self.DANCES_DATASET)
        self.emotions = RecordedMoves(self.EMOTIONS_DATASET)

        # Extract metadata
        self.dance_metadata = self._extract_metadata(self.dances)
        self.emotion_metadata = self._extract_metadata(self.emotions)

        print(f"[ChoreographyContext] Loaded {len(self.dance_metadata)} dances")
        print(f"[ChoreographyContext] Loaded {len(self.emotion_metadata)} emotions")

    def _extract_metadata(self, recorded_moves: RecordedMoves) -> Dict[str, Dict[str, Any]]:
        """
        Extract beat count and metadata for all moves in a library.

        Args:
            recorded_moves: RecordedMoves instance

        Returns:
            Dict mapping move_name -> {beat_count, description}
        """
        metadata = {}
        move_type = "dance" if "dances" in self.DANCES_DATASET else "emotion"

        for move_name in recorded_moves.list_moves():
            try:
                move = recorded_moves.get(move_name)
                beat_count = get_beat_count(move_name, move_type)
                metadata[move_name] = {
                    "beat_count": beat_count,
                    "description": move.description,
                }
            except Exception as e:
                print(f"[ChoreographyContext] Warning: Failed to load {move_name}: {e}")
                continue

        return metadata

    def get_beat_count(self, move_name: str, move_type: str = "emotion") -> int:
        """
        Get beat count for a specific move.

        Args:
            move_name: Name of the move
            move_type: "dance" or "emotion"

        Returns:
            Beat count, or None if not found
        """
        metadata = self.dance_metadata if move_type == "dance" else self.emotion_metadata
        move_info = metadata.get(move_name)
        return move_info["beat_count"] if move_info else None

    def build_moves_context(self) -> Dict[str, Any]:
        """
        Build structured context for LLM with beat counts for BPM-based timing.

        Returns:
            Dict with dances and emotions including beat counts
        """
        return {
            "dances": [
                {
                    "name": name,
                    "beat_count": meta["beat_count"],
                    "description": meta.get("description", "")
                }
                for name, meta in sorted(self.dance_metadata.items())
            ],
            "emotions": [
                {
                    "name": name,
                    "beat_count": meta["beat_count"],
                    "description": meta.get("description", "")
                }
                for name, meta in sorted(self.emotion_metadata.items())
            ]
        }

    def format_for_prompt(self) -> str:
        """
        Format move library with beat counts for BPM-based choreography.

        Returns:
            Formatted string with move names and beat counts
        """
        context = self.build_moves_context()

        # Format dances
        dances_text = "AVAILABLE DANCE MOVES (with beat counts):\n"
        for dance in context["dances"]:
            dances_text += f"  - {dance['name']}: {dance['beat_count']} beats - {dance['description']}\n"

        # Format emotions
        emotions_text = "\nAVAILABLE EMOTIONS (with beat counts):\n"
        for emotion in context["emotions"]:
            emotions_text += f"  - {emotion['name']}: {emotion['beat_count']} beat(s) - {emotion['description']}\n"

        return dances_text + emotions_text

    def calculate_total_duration(self, choreography: List[Dict[str, Any]], bpm: float) -> float:
        """
        Calculate actual total duration using BPM timing formula.

        Formula: Move Duration = cycles × beat_count × (60/BPM)

        Args:
            choreography: List of moves with move_name, move_type, cycles
            bpm: Beats per minute for tempo

        Returns:
            Total duration in seconds
        """
        total = 0.0
        beat_duration = 60.0 / bpm

        for move in choreography:
            move_name = move.get("move_name")
            move_type = move.get("move_type", "emotion")
            cycles = move.get("cycles", 1)

            if move_name == "idle" or move_name == "manual":
                # Idle and manual have custom duration (independent of BPM)
                total += move.get("duration", 1.0)
            else:
                beat_count = self.get_beat_count(move_name, move_type)
                if beat_count:
                    # BPM timing formula
                    duration = cycles * beat_count * beat_duration
                    total += duration
                else:
                    print(f"[ChoreographyContext] Warning: Unknown move {move_name}")

        return total
