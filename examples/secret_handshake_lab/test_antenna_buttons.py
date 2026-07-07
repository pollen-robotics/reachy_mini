"""Unit tests for the base-relative antenna-button press primitive.

The antennas rest at the awake base pose (INIT_ANTENNAS_JOINT_POSITIONS =
[-0.1745, +0.1745] rad, ~10 deg outward each). A "press" is a deviation from
THAT base, not from zero (the fire_nation `|angle| > 0.18` definition only
worked because that game pinned the antennas at [0, 0]). See
docs/superpowers/specs/2026-07-07-antenna-button-handshakes-design.md.

External = deviation in the same sign as the base lean (index 0 negative,
index 1 positive). Internal = the opposite. A release latch (hysteresis)
means a held press counts once; the antenna must fall back near base before
the next press on it counts.

Run:
    python -m pytest examples/secret_handshake_lab/test_antenna_buttons.py -v
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from antenna_buttons import (  # noqa: E402
    AntennaButtonConfig,
    AntennaButtonDetector,
    Direction,
    Press,
)

DT = 0.02  # 50 Hz, same as the daemon control loop
BASE = (-0.1745, 0.1745)  # INIT_ANTENNAS_JOINT_POSITIONS

# Deviations (rad) that clear the 0.18 press threshold with margin.
EXT0 = BASE[0] - 0.30  # antenna 0 pushed further out (more negative)
INT0 = BASE[0] + 0.30  # antenna 0 pulled inward (toward/past center)
EXT1 = BASE[1] + 0.30  # antenna 1 pushed further out (more positive)
INT1 = BASE[1] - 0.30  # antenna 1 pulled inward


def feed(det, samples, goal=BASE):
    """Feed (ant0, ant1) samples tick by tick against a fixed goal; return presses.

    A press is a deviation from the GOAL (commanded target), not from base. The
    default goal is the resting base pose, so the base-relative scenarios below
    read the same as before.
    """
    out = []
    t = 0.0
    for ant0, ant1 in samples:
        out += det.update(t, ant0, ant1, goal[0], goal[1])
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
    """|angle| = 0.18 (the old from-zero threshold) is at base -> no press.

    Antenna 1 held at +0.18 rad absolute deviates only +0.0055 from its base
    of +0.1745. The from-zero definition would fire here; the base-relative
    one must not.
    """
    det = AntennaButtonDetector(AntennaButtonConfig(base=BASE))
    assert feed(det, hold(0.18 * -1, 0.18, 2.0)) == []


def test_held_press_counts_once():
    det = AntennaButtonDetector(AntennaButtonConfig(base=BASE))
    # Antenna 0 stays at base; antenna 1 is held external the whole time.
    presses = feed(det, hold(BASE[0], EXT1, 3.0))
    assert presses == [Press(antenna=1, direction=Direction.EXTERNAL)]


def test_press_release_press_counts_twice():
    det = AntennaButtonDetector(AntennaButtonConfig(base=BASE))
    samples = (
        hold(BASE[0], EXT1, 0.2)  # press
        + hold(BASE[0], BASE[1], 0.2)  # release back to base
        + hold(BASE[0], EXT1, 0.2)  # press again
    )
    presses = feed(det, samples)
    assert presses == [
        Press(antenna=1, direction=Direction.EXTERNAL),
        Press(antenna=1, direction=Direction.EXTERNAL),
    ]


def test_release_needs_hysteresis_not_just_below_press_threshold():
    """A dip to just under the press threshold does not re-arm the latch.

    After a press, the antenna must fall below the RELEASE threshold (0.12
    deviation) before another press counts. A partial release to a 0.15
    deviation, then pressing again, must NOT produce a second press.
    """
    cfg = AntennaButtonConfig(base=BASE, press_threshold_rad=0.18, release_threshold_rad=0.12)
    det = AntennaButtonDetector(cfg)
    partial = BASE[1] + 0.15  # deviation 0.15: below press (0.18), above release (0.12)
    samples = (
        hold(BASE[0], EXT1, 0.2)  # press
        + hold(BASE[0], partial, 0.2)  # partial release (still latched)
        + hold(BASE[0], EXT1, 0.2)  # press again
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
    # Both fire on the SAME tick (first tick of the pressed segment).
    assert len(presses) == 2


def test_multiturn_reading_is_wrapped():
    """An antenna parked a full turn away from base is still at base physically."""
    import math

    det = AntennaButtonDetector(AntennaButtonConfig(base=BASE))
    turn = 2 * math.pi
    # Both antennas at base + one full turn: physically at rest -> no press.
    assert feed(det, hold(BASE[0] + turn, BASE[1] - turn, 2.0)) == []


# ---------------------------------------------------------------------------
# Goal-relative: a press is a deviation from the live commanded target, so it
# works in ANY pose and ignores the robot's own commanded motion.
# ---------------------------------------------------------------------------


def test_no_press_when_present_tracks_a_nonbase_goal():
    det = AntennaButtonDetector(AntennaButtonConfig(base=BASE))
    goal = (0.5, -0.3)  # an arbitrary non-base commanded pose
    out = []
    t = 0.0
    for _ in range(100):
        out += det.update(t, goal[0], goal[1], goal[0], goal[1])
        t += DT
    assert out == []


def test_press_is_deviation_from_goal_not_base():
    det = AntennaButtonDetector(AntennaButtonConfig(base=BASE))
    goal = (0.5, -0.3)
    # Present sits AT the goal (0.5 is far from base but that must not count),
    # then antenna 0 is pushed further external (more negative) than the goal.
    samples = [(goal[0], goal[1])] * 10 + [(goal[0] - 0.30, goal[1])] * 10
    out = []
    t = 0.0
    for a0, a1 in samples:
        out += det.update(t, a0, a1, goal[0], goal[1])
        t += DT
    assert out == [Press(antenna=0, direction=Direction.EXTERNAL)]


def test_moving_goal_perfectly_tracked_is_no_press():
    # The robot ramps the antennas on purpose; present equals goal every tick.
    det = AntennaButtonDetector(AntennaButtonConfig(base=BASE))
    out = []
    t = 0.0
    for k in range(100):
        g0 = BASE[0] - 0.005 * k  # ramps well past the press threshold
        g1 = BASE[1] + 0.005 * k
        out += det.update(t, g0, g1, g0, g1)
        t += DT
    assert out == []


def test_no_press_while_goal_unknown():
    det = AntennaButtonDetector(AntennaButtonConfig(base=BASE))
    out = []
    t = 0.0
    # A big deviation, but with no known goal the subsystem stays inert.
    for _ in range(50):
        out += det.update(t, BASE[0] - 0.5, BASE[1] + 0.5, None, None)
        t += DT
    assert out == []


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
