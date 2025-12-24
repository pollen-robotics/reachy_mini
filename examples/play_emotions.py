#!/usr/bin/env python3
"""
Example on how to play the available emotions on Reachy Mini robot.
"""

from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves


def main():
    try:
        # Initialize emotion library from Hugging Face
        recorded_moves = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")

        # Uncomment this to list all available emotions
        # for move in recorded_moves.list_moves():
        #    print(move)

        # Connect to Reachy Mini
        print("Connecting to Reachy Mini...")
        with ReachyMini(use_sim=False) as reachy:
            print("Connection successful!")

            # Get and play the fear1 emotion
            fear_move = recorded_moves.get("fear1")
            print(f"Playing fear1 emotion: {fear_move.description}")
            reachy.play_move(fear_move, initial_goto_duration=1.0)

            # Get and play the success1 emotion
            success_move = recorded_moves.get("success1")
            print(f"Playing success1 emotion: {success_move.description}")
            reachy.play_move(success_move, initial_goto_duration=1.0)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
