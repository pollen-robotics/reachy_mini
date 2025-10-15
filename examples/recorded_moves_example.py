"""Demonstrate and play all available moves from a dataset for Reachy Mini.

Run :

python3 recorded_moves_example.py -l [dance, emotions]
"""

import argparse

from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMove, RecordedMoves


def main(dataset_path: str) -> None:
    """Connect to Reachy and run the main demonstration loop."""
    recorded_moves = RecordedMoves(dataset_path)

    print("Connecting to Reachy Mini...")
    with ReachyMini(use_sim=False, media_backend="no_media") as reachy:
        print("Connection successful! Starting dance sequence...\n")
        try:
            while True:
                for move_name in recorded_moves.list_moves():
                    move: RecordedMove = recorded_moves.get(move_name)
                    print(f"Playing move: {move_name}: {move.description}\n")
                    # print(f"params: {move.move_params}")
                    reachy.play_move(move, initial_goto_duration=1.0)

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
    main(dataset_path)
