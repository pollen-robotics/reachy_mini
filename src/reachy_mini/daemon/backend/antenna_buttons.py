"""Base-relative antenna-button press primitive for the torque-ON handshakes.

Pure, dependency-free, fast: this runs every tick on the 50 Hz control loop
(CM4), so it is a handful of float compares, no allocation beyond the small
returned list, no I/O.

WHY GOAL-RELATIVE (the robust definition)

A press is a deviation of the PRESENT antenna angle from its GOAL (the
commanded target), not from zero and not from a fixed base:

    dev_i = wrap_to_pi(present_i - goal_i)      # wrap: encoders are multi-turn
    ext_i = sign(base_i)                        # each antenna's external side
    external press:  dev_i * ext_i >  press_threshold
    internal press:  dev_i * ext_i < -press_threshold

This is robust in ANY pose and immune to the robot's own commanded motion:
when the robot moves the antennas on purpose the present angle tracks the
goal, so dev stays ~0 and nothing triggers; only an EXTERNAL push (present
diverges from the commanded goal) registers as a press. If the goal is
unknown (None), the subsystem stays inert (no presses).

`base` only fixes each antenna's external DIRECTION: index 0's external is
negative, index 1's is positive (the awake-base lean signs). It is no longer
the deviation reference. The left/right labelling (fire_nation `[0]=right`,
collision code `[0]=left`) does NOT affect detection; it only matters when
mapping the coded sequences, confirmed on the robot.

RELEASE LATCH (hysteresis)

After a press is counted on an antenna, that antenna must fall back below the
RELEASE threshold (< press threshold, a hysteresis band near base) before the
next press on it counts. Button presses are deliberate and well separated, so
a latch is correct here (unlike the collision path, where a release latch was
tried and reverted because fast knocks are only 2-3 ticks apart).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import NamedTuple

# Mirror of reachy_mini INIT_ANTENNAS_JOINT_POSITIONS, kept local so this
# module stays import-free and trivially testable (same pattern as the
# collision module's local _SLEEP_HEAD_POSE copy). The live daemon passes the
# authoritative constant into the config.
DEFAULT_BASE = (-0.1745, 0.1745)

_TWO_PI = 2.0 * math.pi


def _wrap_pi(rad: float) -> float:
    """Wrap a multi-turn angle to [-pi, pi)."""
    return (rad + math.pi) % _TWO_PI - math.pi


class Direction(enum.Enum):
    """Which way an antenna was pressed, relative to its base lean."""

    EXTERNAL = "external"  # deviation in the same sign as the base lean
    INTERNAL = "internal"  # deviation toward / past center


class Press(NamedTuple):
    """A press that STARTED this tick: which antenna, which direction."""

    antenna: int
    direction: Direction


@dataclass(frozen=True)
class AntennaButtonConfig:
    """Tunables for the press primitive (all radians)."""

    base: tuple[float, float] = DEFAULT_BASE  # only fixes each external direction
    press_threshold_rad: float = 0.18  # deviation from goal to count a press
    release_threshold_rad: float = 0.12  # must fall below this to re-arm


class AntennaButtonDetector:
    """Turns raw antenna angles into base-relative press edge events.

    Feed present antenna positions (radians) with the sample time every tick;
    `update()` returns the list of presses that STARTED this tick (0, 1, or 2
    -- both antennas can press on the same tick, e.g. the symmetric WiFi
    code). `dev` exposes the current signed-external deviations for debug.
    """

    def __init__(self, config: AntennaButtonConfig | None = None) -> None:
        """Build the detector; precompute each antenna's external direction."""
        self.cfg = config or AntennaButtonConfig()
        self._ext = (
            _sign(self.cfg.base[0]),
            _sign(self.cfg.base[1]),
        )
        # Per-antenna latch: True while a counted press is still held (waiting
        # for a release below the release threshold to re-arm).
        self._latched = [False, False]
        # Signed-external deviation per antenna (public, for the live probe).
        self.dev = [0.0, 0.0]

    def reset(self) -> None:
        """Forget the latch state and cached deviations."""
        self._latched = [False, False]
        self.dev = [0.0, 0.0]

    def update(
        self,
        t: float,
        ant0: float,
        ant1: float,
        goal0: float | None = None,
        goal1: float | None = None,
    ) -> list[Press]:
        """Return the presses that started this tick (edge-detected).

        A press is a deviation of the present angle from the commanded goal.
        With no goal (goal0/goal1 is None) the subsystem stays inert.
        """
        if goal0 is None or goal1 is None:
            return []
        goals = (goal0, goal1)
        presses: list[Press] = []
        for i, angle in enumerate((ant0, ant1)):
            dev = _wrap_pi(angle - goals[i])
            signed = dev * self._ext[i]  # >0 external, <0 internal
            self.dev[i] = signed
            if self._latched[i]:
                if abs(signed) < self.cfg.release_threshold_rad:
                    self._latched[i] = False
                continue
            if signed > self.cfg.press_threshold_rad:
                self._latched[i] = True
                presses.append(Press(antenna=i, direction=Direction.EXTERNAL))
            elif signed < -self.cfg.press_threshold_rad:
                self._latched[i] = True
                presses.append(Press(antenna=i, direction=Direction.INTERNAL))
        return presses


def _sign(x: float) -> float:
    """Sign of the base lean; a zero base defaults to +1 (external = positive)."""
    return -1.0 if x < 0.0 else 1.0
