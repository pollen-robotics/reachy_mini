import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.motion.dance_move import DanceMove
from reachy_mini.motion.collection.dance import AVAILABLE_MOVES

possible_moves = list(AVAILABLE_MOVES.keys())

with ReachyMini() as reachy:
    while True:
        for move_name in possible_moves:
            move = DanceMove(move_name)
            print(f"Playing move: {move_name}: {move.move_metadata['description']}\n")
            # print(f"params: {move.move_params}")
            move.play_on(reachy, repeat=1)
            print(f"Move {move_name} completed.")
