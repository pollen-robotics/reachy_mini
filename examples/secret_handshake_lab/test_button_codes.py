"""Unit tests for the torque-ON antenna-button code state machine.

Three coded sequences of antenna-button presses, each triggering one action
(see docs/superpowers/specs/2026-07-07-antenna-button-handshakes-design.md):

  WiFi        BOTH external together, then BOTH internal together
  Torque off  3x left-external, then 2x right-external
  Emotion     left-ext, right-ext, left-ext, right-ext (alternating)

Rules: external-only for the single-press codes, <1 s between presses/moves
or the sequence resets, a small simultaneity window for the paired WiFi move,
armed only while torque is ON.

The machine consumes the Press edge events emitted by AntennaButtonDetector,
so these tests feed press events tick by tick exactly as the daemon will.

Run:
    python -m pytest examples/secret_handshake_lab/test_button_codes.py -v
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from antenna_buttons import Direction, Press  # noqa: E402
from button_codes import ButtonCodeConfig, ButtonCodeMachine, CodeEvent  # noqa: E402

DT = 0.02  # 50 Hz
L, R = 0, 1
EXT, INT = Direction.EXTERNAL, Direction.INTERNAL


def feed(m, ticks, torque_on=True):
    """Feed per-tick press lists; return the CodeEvents emitted."""
    out = []
    t = 0.0
    for presses in ticks:
        ev = m.update(t, presses, torque_on)
        if ev is not None:
            out.append(ev)
        t += DT
    return out


def one(antenna, direction, gap=6):
    """A single press on one antenna, followed by a release gap (ticks)."""
    return [[Press(antenna, direction)]] + [[] for _ in range(gap)]


def sym(direction, gap=6):
    """Both antennas pressed the same direction on the same tick (a move)."""
    return [[Press(0, direction), Press(1, direction)]] + [[] for _ in range(gap)]


def empties(n):
    return [[] for _ in range(n)]


def test_wifi_code_fires_and_nothing_else():
    m = ButtonCodeMachine(ButtonCodeConfig())
    assert feed(m, sym(EXT) + sym(INT)) == [CodeEvent.WIFI]


def test_torque_off_code_fires():
    m = ButtonCodeMachine(ButtonCodeConfig())
    ticks = one(L, EXT) + one(L, EXT) + one(L, EXT) + one(R, EXT) + one(R, EXT)
    assert feed(m, ticks) == [CodeEvent.TORQUE_OFF]


def test_emotion_code_fires():
    m = ButtonCodeMachine(ButtonCodeConfig())
    ticks = one(L, EXT) + one(R, EXT) + one(L, EXT) + one(R, EXT)
    assert feed(m, ticks) == [CodeEvent.EMOTION]


def test_wrong_first_press_fires_nothing():
    m = ButtonCodeMachine(ButtonCodeConfig())
    # right-external as a single press starts none of the codes.
    assert feed(m, one(R, EXT) + one(R, EXT) + one(R, EXT)) == []


def test_gap_over_one_second_resets_sequence():
    m = ButtonCodeMachine(ButtonCodeConfig())
    # 3 left-external, then a >1 s pause, then 2 right-external: the reset
    # means the torque-off code never completes.
    ticks = one(L, EXT) + one(L, EXT) + one(L, EXT) + empties(60) + one(R, EXT) + one(R, EXT)
    assert feed(m, ticks) == []


def test_internal_single_presses_start_no_code():
    m = ButtonCodeMachine(ButtonCodeConfig())
    ticks = one(L, INT) + one(L, INT) + one(L, INT) + one(R, INT) + one(R, INT)
    assert feed(m, ticks) == []


def test_disarmed_when_torque_off():
    m = ButtonCodeMachine(ButtonCodeConfig())
    ticks = one(L, EXT) + one(R, EXT) + one(L, EXT) + one(R, EXT)
    assert feed(m, ticks, torque_on=False) == []


def test_wifi_needs_both_antennas_within_the_window():
    m = ButtonCodeMachine(ButtonCodeConfig())
    # Left then right external but far apart -> two singles, not a paired
    # move -> WiFi never completes even with a following internal move.
    ticks = (
        [[Press(0, EXT)]]
        + empties(6)
        + [[Press(1, EXT)]]
        + empties(6)
        + sym(INT)
    )
    assert CodeEvent.WIFI not in feed(m, ticks)


def test_wifi_needs_internal_move_second():
    m = ButtonCodeMachine(ButtonCodeConfig())
    assert feed(m, sym(EXT) + sym(EXT)) == []


def test_stray_press_before_wifi_still_fires():
    m = ButtonCodeMachine(ButtonCodeConfig())
    # A stray single external press, then the full WiFi move: the machine
    # slides past the stray and still recognises WiFi.
    ticks = one(R, EXT) + sym(EXT) + sym(INT)
    assert feed(m, ticks) == [CodeEvent.WIFI]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
