"""Unit tests for the daemon torque-ON antenna-button code state machine.

Mirror of examples/secret_handshake_lab/test_button_codes.py against the
shipped module. See
docs/superpowers/specs/2026-07-07-antenna-button-handshakes-design.md.
"""

from reachy_mini.daemon.backend.antenna_buttons import Direction, Press
from reachy_mini.daemon.backend.button_codes import (
    ButtonCodeConfig,
    ButtonCodeMachine,
    CodeEvent,
)

DT = 0.02
L, R = 0, 1
EXT, INT = Direction.EXTERNAL, Direction.INTERNAL


def feed(m, ticks, torque_on=True):
    out = []
    t = 0.0
    for presses in ticks:
        ev = m.update(t, presses, torque_on)
        if ev is not None:
            out.append(ev)
        t += DT
    return out


def one(antenna, direction, gap=6):
    return [[Press(antenna, direction)]] + [[] for _ in range(gap)]


def sym(direction, gap=6):
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
    assert feed(m, one(R, EXT) + one(R, EXT) + one(R, EXT)) == []


def test_gap_over_one_second_resets_sequence():
    m = ButtonCodeMachine(ButtonCodeConfig())
    ticks = (
        one(L, EXT) + one(L, EXT) + one(L, EXT) + empties(60) + one(R, EXT) + one(R, EXT)
    )
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
    ticks = [[Press(0, EXT)]] + empties(6) + [[Press(1, EXT)]] + empties(6) + sym(INT)
    assert CodeEvent.WIFI not in feed(m, ticks)


def test_wifi_needs_internal_move_second():
    m = ButtonCodeMachine(ButtonCodeConfig())
    assert feed(m, sym(EXT) + sym(EXT)) == []


def test_stray_press_before_wifi_still_fires():
    m = ButtonCodeMachine(ButtonCodeConfig())
    ticks = one(R, EXT) + sym(EXT) + sym(INT)
    assert feed(m, ticks) == [CodeEvent.WIFI]
