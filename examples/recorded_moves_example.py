"""Demonstrate and play all available moves from a dataset for Reachy Mini.

Run :

python3 recorded_moves_example.py -l [dance, emotions]
"""

import argparse

from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMove, RecordedMoves

# Keep compatibility with the original library flag while allowing custom datasets.
LIBRARY_DATASETS = {
    "dance": "pollen-robotics/reachy-mini-dances-library",
    "emotions": "pollen-robotics/reachy-mini-emotions-library",
}


def main(dataset_path: str) -> None:
    """Connect to Reachy and run the main demonstration loop."""
    recorded_moves = RecordedMoves(dataset_path)

    print("Connecting to Reachy Mini...")
    with ReachyMini() as reachy:
        print("Connection successful! Starting dance sequence...\n")
        try:
            while True:
                for move_name in recorded_moves.list_moves():
                    move: RecordedMove = recorded_moves.get(move_name)
                    print(f"Playing move: {move_name}: {move.description}\n")
                    reachy.play_move(move, initial_goto_duration=1.0)

        except KeyboardInterrupt:
            print("\n Sequence interrupted by user. Shutting down.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstrate and play all available dance moves for Reachy Mini."
    )
    parser.add_argument(
        "-l",
        "--library",
        type=str,
        default="dance",
        choices=sorted(LIBRARY_DATASETS.keys()),
        help="Pick one of the original libraries (default: dance).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Local path or HF dataset id. Overrides --library when provided.",
    )
    args = parser.parse_args()
    dataset_path = args.dataset or LIBRARY_DATASETS[args.library]
    main(dataset_path)
