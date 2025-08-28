from reachy_mini import ReachyMini  # noqa: D100
from reachy_mini.motion.recorded import RecordedMoves

with ReachyMini() as mini:
    recorded_moves = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
    print(recorded_moves.list_moves())
    recorded_moves.get("welcoming1").play_on(mini, repeat=1, start_goto=True)
    recorded_moves.get("yes1").play_on(mini, repeat=1, start_goto=True)
    recorded_moves.get("tired1").play_on(mini, repeat=1, start_goto=True)
    recorded_moves.get("sad1").play_on(mini, repeat=1, start_goto=True)
