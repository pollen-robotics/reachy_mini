"""Base-relative antenna-button press primitive for the torque-ON handshakes.

Pure, dependency-free, fast: this runs every tick on the 50 Hz control loop
(CM4), so it is a handful of float compares, no allocation beyond the small
returned list, no I/O.

WHY BASE-RELATIVE (the correction that motivated the redesign)

The antennas now rest at the awake base pose
`INIT_ANTENNAS_JOINT_POSITIONS = [-0.1745, +0.1745]` rad (~10 deg outward
each). The `fire_nation_attacked` button definition (`|angle| > 0.18 rad`)
measures deviation from ZERO and only works there because that game pins the
antennas at [0, 0]. At the awake base pose `|angle|` is already ~0.1745 at
rest, right at that threshold, so a from-zero definition would read the
resting antennas as pressed.

A PRESS IS THEREFORE A DEVIATION FROM THE BASE POSITION, not from zero:

    dev_i = wrap_to_pi(angle_i - base_i)        # wrap: encoders are multi-turn
    ext_i = sign(base_i)                        # the base lean IS "external"
    external press:  dev_i * ext_i >  press_threshold
    internal press:  dev_i * ext_i < -press_threshold

Index 0's external direction is negative, index 1's is positive (the base
signs). The left/right labelling (fire_nation `[0]=right`, collision code
`[0]=left`) does NOT affect detection; it only matters when mapping the coded
sequences, which is confirmed on the robot.

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
    EXTERNAL = "external"  # deviation in the same sign as the base lean
    INTERNAL = "internal"  # deviation toward / past center


class Press(NamedTuple):
    """A press that STARTED this tick: which antenna, which direction."""

    antenna: int
    direction: Direction


@dataclass(frozen=True)
class AntennaButtonConfig:
    """Tunables for the press primitive (all radians)."""

    base: tuple[float, float] = DEFAULT_BASE
    press_threshold_rad: float = 0.18  # deviation from base to count a press
    release_threshold_rad: float = 0.12  # must fall below this to re-arm


class AntennaButtonDetector:
    """Turns raw antenna angles into base-relative press edge events.

    Feed present antenna positions (radians) with the sample time every tick;
    `update()` returns the list of presses that STARTED this tick (0, 1, or 2
    -- both antennas can press on the same tick, e.g. the symmetric WiFi
    code). `dev` exposes the current signed-external deviations for debug.
    """

    def __init__(self, config: AntennaButtonConfig | None = None) -> None:
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
        self._latched = [False, False]
        self.dev = [0.0, 0.0]

    def update(self, t: float, ant0: float, ant1: float) -> list[Press]:
        """Return the presses that started this tick (edge-detected)."""
        presses: list[Press] = []
        for i, angle in enumerate((ant0, ant1)):
            dev = _wrap_pi(angle - self.cfg.base[i])
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
