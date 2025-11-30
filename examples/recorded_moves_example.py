"""Demonstrate and play all available moves from a dataset for Reachy Mini.

Run :

python3 recorded_moves_example.py -l [dance, emotions]
"""

import argparse
from pathlib import Path

from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMove, RecordedMoves


def main(dataset_path: str, library_type: str) -> None:
    """Connect to Reachy and run the main demonstration loop."""
    recorded_moves = RecordedMoves(dataset_path)

    print("Connecting to Reachy Mini...")
    # Use default media backend for emotions to enable sound, no_media for dances
    media_backend = "default" if library_type == "emotions" else "no_media"
    with ReachyMini(use_sim=False, media_backend=media_backend) as reachy:
        print(f"Connection successful! Starting {library_type} sequence...\n")
        try:
            while True:
                for move_name in recorded_moves.list_moves():
                    move: RecordedMove = recorded_moves.get(move_name)
                    print(f"Playing move: {move_name}: {move.description}")
                    
                    # Play sound if available and file exists (for emotions library)
                    if move.sound_path and Path(move.sound_path).exists():
                        reachy.media.play_sound(move.sound_path)
                    
                    reachy.play_move(move, initial_goto_duration=1.0)
                    print()

        except KeyboardInterrupt:
            print("\n Sequence interrupted by user. Shutting down.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstrate and play all available dance moves for Reachy Mini."
    )
    parser.add_argument(
        "-l", "--library", type=str, default="dance", choices=["dance", "emotions"]
    )
    args = parser.parse_args()

    dataset_path = (
        "pollen-robotics/reachy-mini-dances-library"
        if args.library == "dance"
        else "pollen-robotics/reachy-mini-emotions-library"
    )
    main(dataset_path, args.library)
