"""The bundled handshake emotion move must ship and load as a RecordedMove.

Guards the asset the torque-ON emotion button code plays (see
docs/superpowers/specs/2026-07-07-antenna-button-handshakes-design.md). If the
file is missing or malformed the daemon skips the emotion gracefully, but it
should ship valid, so this is a hard check.
"""

import json
from importlib import resources

from reachy_mini.motion.recorded_move import RecordedMove


def test_bundled_excited_move_loads_and_is_short():
    base = resources.files("reachy_mini") / "assets" / "handshake_moves"
    move_file = base / "excited.json"
    assert move_file.is_file(), "excited.json must ship under assets/handshake_moves"

    data = json.loads(move_file.read_text())
    move = RecordedMove(data, None)

    assert move.duration > 0.0
    # A fixed SHORT excited move (some library moves are very long).
    assert move.duration < 6.0
    # The move can be sampled across its span without raising (evaluate()
    # rejects t at/after the final timestamp by design, so stop just short).
    move.evaluate(0.0)
    move.evaluate(move.duration * 0.99)
