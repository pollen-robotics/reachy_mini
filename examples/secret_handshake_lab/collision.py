"""Antenna collision detection primitive for the secret handshake.

Pure, dependency-free, fast. This is the ONE thing that will eventually be
called every tick from the 50 Hz control loop, so it must stay trivial:
a handful of float ops, no allocation, no I/O.

THE LAW (validated on RemiFabre/secret-handshake, see analyze_recordings.py):

The antennas are floppy friction-fit parts. Their angles mean nothing in
absolute terms: they rest wherever they were last left (one robot rested at
diff=+0.97 where another rested at -0.35), and the same angle pair can occur
both touching and not touching (paused mid-slide vs parked crossed). So no
instantaneous function of the angles can define contact.

What IS distinctive is COUPLED MOTION: when the antennas touch, moving one
moves the other. With v0, v1 the angular velocities, define

    m = min(|v0|, |v1|)        # coupled speed, rad/s

    rest / parked / one antenna moved alone :  m ~ 0
    touching and sliding (gentle rub)       :  m ~ 0.3 .. 1.2, sustained
    collision (audible knock)               :  m spikes to 4 .. 9

Measured on the recordings: every audible knock is an m-spike > 4; thirty
seconds of slide data never exceed 1.2. Threshold at 2.0 with a refractory
and each knock counts exactly once, regardless of where the antennas hang,
which antenna is swung, or which robot it is.

This module deliberately does NOT know about the head pose, torque state,
rhythm, or the state machine above it. It answers: "did a collision just
happen?" (update() returns True) and "are the antennas being moved together
right now?" (.coupled / .coupled_speed).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CollisionConfig:
    # Velocity is measured across the last `vel_span` sample steps
    # (2 steps = 40 ms at 50 Hz): long enough to bridge a stale read,
    # short enough to catch a knock spike.
    vel_span: int = 2
    # Coupled speed above this is a knock (recordings: knocks 4..9, slides
    # never exceed 1.2, so 2.0 sits in the middle of a wide margin).
    knock_on: float = 2.0
    # One collision = one count even if it bounces (recordings show knock
    # spacing ~0.4 s when tapping naturally).
    knock_refractory_s: float = 0.25
    # Coupled speed above this means the antennas are touching and being
    # moved together (the rub/slide gesture).
    couple_on: float = 0.25


class CollisionDetector:
    """Stateful coupled-motion detector for antenna collisions.

    Feed it present antenna positions (with the sample time) every tick:
    update() returns True exactly once per knock. `coupled` / `coupled_speed`
    expose the sustained-contact signal for the rub gesture and debugging.
    """

    def __init__(self, config: CollisionConfig | None = None) -> None:
        self.cfg = config or CollisionConfig()
        self.coupled: bool = False
        self.coupled_speed: float = 0.0
        self._hist: list[tuple[float, float, float]] = []  # (t, ant0, ant1)
        self._prev_m: float = 0.0
        self._last_knock_t: float = -1e9

    def reset(self) -> None:
        self.coupled = False
        self.coupled_speed = 0.0
        self._hist.clear()
        self._prev_m = 0.0
        self._last_knock_t = -1e9

    def update(self, t: float, ant0: float, ant1: float) -> bool:
        """Return True exactly on the tick a new collision (knock) is seen."""
        self._hist.append((t, ant0, ant1))
        if len(self._hist) <= self.cfg.vel_span:
            return False
        if len(self._hist) > self.cfg.vel_span + 1:
            self._hist.pop(0)

        t0, x0, y0 = self._hist[0]
        dt = t - t0
        if dt <= 0.0:
            return False
        v0 = (ant0 - x0) / dt
        v1 = (ant1 - y0) / dt
        m = min(abs(v0), abs(v1))

        self.coupled_speed = m
        self.coupled = m > self.cfg.couple_on

        knock = (
            m > self.cfg.knock_on
            and self._prev_m <= self.cfg.knock_on
            and t - self._last_knock_t > self.cfg.knock_refractory_s
        )
        if knock:
            self._last_knock_t = t
        self._prev_m = m
        return knock
