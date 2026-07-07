"""Unit tests for the daemon base-relative antenna-button press primitive.

Mirror of examples/secret_handshake_lab/test_antenna_buttons.py against the
shipped module. A press is a deviation from the awake base pose
(INIT_ANTENNAS_JOINT_POSITIONS), not from zero. See
docs/superpowers/specs/2026-07-07-antenna-button-handshakes-design.md.
"""

import math

from reachy_mini.daemon.backend.antenna_buttons import (
    AntennaButtonConfig,
    AntennaButtonDetector,
    Direction,
    Press,
)

DT = 0.02
BASE = (-0.1745, 0.1745)

EXT0 = BASE[0] - 0.30
INT0 = BASE[0] + 0.30
EXT1 = BASE[1] + 0.30
INT1 = BASE[1] - 0.30


def feed(det, samples):
    out = []
    t = 0.0
    for ant0, ant1 in samples:
        out += det.update(t, ant0, ant1)
        t += DT
    return out


def hold(ant0, ant1, seconds):
    return [(ant0, ant1)] * max(1, round(seconds / DT))


def test_rest_is_no_press():
    det = AntennaButtonDetector(AntennaButtonConfig(base=BASE))
    assert feed(det, hold(BASE[0], BASE[1], 2.0)) == []


def test_external_press_antenna0_is_negative_deviation():
    det = AntennaButtonDetector(AntennaButtonConfig(base=BASE))
    presses = feed(det, hold(BASE[0], BASE[1], 0.2) + hold(EXT0, BASE[1], 0.2))
    assert presses == [Press(antenna=0, direction=Direction.EXTERNAL)]


def test_external_press_antenna1_is_positive_deviation():
    det = AntennaButtonDetector(AntennaButtonConfig(base=BASE))
    presses = feed(det, hold(BASE[0], BASE[1], 0.2) + hold(BASE[0], EXT1, 0.2))
    assert presses == [Press(antenna=1, direction=Direction.EXTERNAL)]


def test_internal_press_antenna0():
    det = AntennaButtonDetector(AntennaButtonConfig(base=BASE))
    presses = feed(det, hold(BASE[0], BASE[1], 0.2) + hold(INT0, BASE[1], 0.2))
    assert presses == [Press(antenna=0, direction=Direction.INTERNAL)]


def test_internal_press_antenna1():
    det = AntennaButtonDetector(AntennaButtonConfig(base=BASE))
    presses = feed(det, hold(BASE[0], BASE[1], 0.2) + hold(BASE[0], INT1, 0.2))
    assert presses == [Press(antenna=1, direction=Direction.INTERNAL)]


def test_absolute_angle_near_threshold_but_at_base_is_not_a_press():
    det = AntennaButtonDetector(AntennaButtonConfig(base=BASE))
    assert feed(det, hold(0.18 * -1, 0.18, 2.0)) == []


def test_held_press_counts_once():
    det = AntennaButtonDetector(AntennaButtonConfig(base=BASE))
    presses = feed(det, hold(BASE[0], EXT1, 3.0))
    assert presses == [Press(antenna=1, direction=Direction.EXTERNAL)]


def test_press_release_press_counts_twice():
    det = AntennaButtonDetector(AntennaButtonConfig(base=BASE))
    samples = (
        hold(BASE[0], EXT1, 0.2) + hold(BASE[0], BASE[1], 0.2) + hold(BASE[0], EXT1, 0.2)
    )
    presses = feed(det, samples)
    assert presses == [
        Press(antenna=1, direction=Direction.EXTERNAL),
        Press(antenna=1, direction=Direction.EXTERNAL),
    ]


def test_release_needs_hysteresis_not_just_below_press_threshold():
    cfg = AntennaButtonConfig(
        base=BASE, press_threshold_rad=0.18, release_threshold_rad=0.12
    )
    det = AntennaButtonDetector(cfg)
    partial = BASE[1] + 0.15
    samples = (
        hold(BASE[0], EXT1, 0.2)
        + hold(BASE[0], partial, 0.2)
        + hold(BASE[0], EXT1, 0.2)
    )
    presses = feed(det, samples)
    assert presses == [Press(antenna=1, direction=Direction.EXTERNAL)]


def test_both_antennas_press_on_same_tick():
    det = AntennaButtonDetector(AntennaButtonConfig(base=BASE))
    presses = feed(det, hold(BASE[0], BASE[1], 0.2) + hold(EXT0, EXT1, 0.2))
    assert set(presses) == {
        Press(antenna=0, direction=Direction.EXTERNAL),
        Press(antenna=1, direction=Direction.EXTERNAL),
    }
    assert len(presses) == 2


def test_multiturn_reading_is_wrapped():
    det = AntennaButtonDetector(AntennaButtonConfig(base=BASE))
    turn = 2 * math.pi
    assert feed(det, hold(BASE[0] + turn, BASE[1] - turn, 2.0)) == []
