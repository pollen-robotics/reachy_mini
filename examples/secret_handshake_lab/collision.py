"""Antenna collision detection primitive for the secret handshake.

Pure, dependency-free, fast. This is the ONE thing that will eventually be
called every tick from the 50 Hz control loop, so it must stay trivial:
a handful of float compares, no allocation, no I/O.

THE LAW (v3, geometric; measured by Remi by hand on 2 robots, in degrees):

The antennas collide "at the center" (along the y = 0 plane) when both:

    1) l_ant + r_ant in [-7, -3]    (~= 0, with a slight consistent
                                     asymmetry seen on both robots)
    2) l_ant        in [20, 150]

widened by a margin: a 4 deg margin turns the sum band into [-9, -1].
l_ant is antenna index 0, r_ant is index 1, as returned by
`get_present_antenna_joint_positions()` (radians; converted internally).

A collision EVENT is entering that region. A firm press flexes the antennas
through the band (sum shoots past it, up to +35..+73 deg in the recordings)
and back through it on release, so a short refractory merges the double
crossing into one count. Validated on RemiFabre/secret-handshake: exactly 3
collisions on each of the 4 recorded gestures, at the audible knock times;
the approach ("bring together") does not count because condition 2 fails
while the left antenna is still below 20 deg.

Condition 2 also makes rest inert: at a typical floppy rest (l=-12, r=+10)
the sum (-2) is inside the band but l is far below 20 deg.

This module deliberately does NOT know about the head pose, torque state,
rhythm, or the state machine above it. It answers: "did a collision just
happen?" (update() returns True) and "are the antennas in the collision
region right now?" (.in_collision, used for the gentle-hold gesture).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_RAD2DEG = 180.0 / math.pi


def _wrap_deg(deg: float) -> float:
    """Wrap a multi-turn angle to [-180, 180).

    The antenna encoders are multi-turn: handling the floppy antennas can
    park them whole turns away from the calibrated zero (seen live at rest:
    l=-340, r=+334, physically l=+20, r=-26). The law is about the physical
    angle, so it must apply to the wrapped reading.
    """
    return (deg + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class CollisionConfig:
    # Center-collision geometry, in DEGREES. Remi measured sum in [-7, -3]
    # on 2 robots; live testing on a 3rd (different batch) showed many real
    # collisions right at the top edge, so the top was relaxed to -2: with
    # the margin the band becomes [-9, 0], 0 being the theoretical
    # symmetric contact point. Do NOT lower the bottom: the crossed-parked
    # state (antennas slid past each other, resting) sits at sum ~ -11.
    sum_min_deg: float = -7.0
    sum_max_deg: float = -2.0
    l_min_deg: float = 20.0
    l_max_deg: float = 150.0
    # Widens the sum band by margin/2 on each side: 4 -> [-9, -1].
    margin_deg: float = 4.0
    # One knock crosses the band twice (in, then back out through it while
    # releasing); the refractory merges that into one collision. Natural
    # knock spacing in the recordings is ~0.4 s, so 0.25 s sits safely below.
    refractory_s: float = 0.25


class CollisionDetector:
    """Stateful edge detector for the geometric collision region.

    Feed it present antenna positions (radians) with the sample time every
    tick: update() returns True exactly once per collision. `in_collision`
    exposes the current region membership (for the hold gesture and debug);
    `sum_deg` / `l_deg` expose the numbers behind the decision.
    """

    def __init__(self, config: CollisionConfig | None = None) -> None:
        self.cfg = config or CollisionConfig()
        half = self.cfg.margin_deg / 2.0
        # Margined band bounds, public for display (e.g. [-9, -1] by default).
        self.sum_lo_deg = self.cfg.sum_min_deg - half
        self.sum_hi_deg = self.cfg.sum_max_deg + half
        self.in_collision: bool = False
        self.sum_deg: float = 0.0
        self.l_deg: float = 0.0
        self._last_onset_t: float = -1e9

    def reset(self) -> None:
        self.in_collision = False
        self._last_onset_t = -1e9

    def update(self, t: float, ant0: float, ant1: float) -> bool:
        """Return True exactly on the tick a new collision is seen."""
        l_deg = _wrap_deg(ant0 * _RAD2DEG)
        # Wrap the sum too: l and r wrapped independently can land one full
        # turn apart (e.g. l=+170, r=-190 -> wrapped r=+170, sum=+340).
        sum_deg = _wrap_deg(l_deg + _wrap_deg(ant1 * _RAD2DEG))
        self.l_deg = l_deg
        self.sum_deg = sum_deg

        inside = (
            self.sum_lo_deg <= sum_deg <= self.sum_hi_deg
            and self.cfg.l_min_deg <= l_deg <= self.cfg.l_max_deg
        )
        onset = (
            inside
            and not self.in_collision
            and t - self._last_onset_t > self.cfg.refractory_s
        )
        self.in_collision = inside
        if onset:
            self._last_onset_t = t
        return onset
