from reachy_mini import ReachyMini  # noqa: D100
from reachy_mini.motion.recorded_move import RecordedMoves

with ReachyMini() as mini:
    recorded_moves = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
    print(recorded_moves.list_moves())

    mini.play_move(recorded_moves.get("welcoming1"))
    mini.play_move(recorded_moves.get("yes1"))
    mini.play_move(recorded_moves.get("tired1"))
    mini.play_move(recorded_moves.get("sad1"))
