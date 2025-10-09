"""Demonstrate and play all available dance moves for Reachy Mini.

----------------------------------

This script iterates through the `AVAILABLE_MOVES` dictionary and executes
each move once on the connected Reachy Mini robot in an infinite loop.
"""

from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMove, RecordedMoves


def main() -> None:
    """Connect to Reachy and run the main demonstration loop."""
    recorded_moves = RecordedMoves("pollen-robotics/reachy-mini-dances-library")

    print("Connecting to Reachy Mini...")
    with ReachyMini(use_sim=False) as reachy:
        print("Connection successful! Starting dance sequence...\n")
        try:
            while True:
                # recorded_moves.moves is a dict, iterate inside the dict:

                for move_name in recorded_moves.moves:
                    move: RecordedMove = recorded_moves.get(move_name)
                    print(f"Playing move: {move_name}: {move.description}\n")
                    # print(f"params: {move.move_params}")
                    reachy.play_move(move, initial_goto_duration=1.0)

        except KeyboardInterrupt:
            print("\nDance sequence interrupted by user. Shutting down.")


if __name__ == "__main__":
    main()
