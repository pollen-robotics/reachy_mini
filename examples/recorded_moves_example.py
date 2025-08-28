from reachy_mini import ReachyMini
from reachy_mini.motion.recorded import RecordedMoves

with ReachyMini() as mini:
    recorded_moves = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
    recorded_moves.get("welcoming1").play_on(mini, repeat=1, start_goto=True)
    recorded_moves.get("yes1").play_on(mini, repeat=1, start_goto=True)
    recorded_moves.get("tired1").play_on(mini, repeat=1, start_goto=True)
    recorded_moves.get("sad1").play_on(mini, repeat=1, start_goto=True)
