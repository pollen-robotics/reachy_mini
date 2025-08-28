"""Demonstrate and play all available dance moves for Reachy Mini.

----------------------------------

This script iterates through the `AVAILABLE_MOVES` dictionary and executes
each move once on the connected Reachy Mini robot in an infinite loop.
"""

from reachy_mini import ReachyMini
from reachy_mini.motion.dance.collection.dance import AVAILABLE_MOVES
from reachy_mini.motion.dance import DanceMove


def main() -> None:
    """Connect to Reachy and run the main demonstration loop."""
    possible_moves: list[str] = list(AVAILABLE_MOVES.keys())

    print("Connecting to Reachy Mini...")
    with ReachyMini() as reachy:
        print("Connection successful! Starting dance sequence...\n")
        try:
            while True:
                for move_name in possible_moves:
                    move: DanceMove = DanceMove(move_name)
                    print(
                        f"Playing move: {move_name}: "
                        f"{move.move_metadata['description']}\n"
                    )
                    # print(f"params: {move.move_params}")
                    move.play_on(reachy, repeat=1)

        except KeyboardInterrupt:
            print("\nDance sequence interrupted by user. Shutting down.")


if __name__ == "__main__":
    main()
