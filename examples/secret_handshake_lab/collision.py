"""Antenna collision detection primitive for the secret handshake.

Pure, dependency-free, fast. This is the ONE thing that will eventually be
called every tick from the 50 Hz control loop, so it must stay trivial:
a couple of float compares and a tiny state machine, no allocation, no I/O.

Definition of "collision / contact" (derived from real recorded data, see
replay_validate.py and the design spec):

    Antennas present positions come as [ant0, ant1] (index 0 = left,
    index 1 = right), in radians.

    At natural torque-OFF rest the antennas are slightly splayed outward:
        ant0 ~ -0.18   ant1 ~ +0.17   ->  diff = ant0 - ant1 ~ -0.35
    When the user brings them together at the center they converge and press:
        contact begins as diff crosses ~0, firm contact/flex reaches diff 2..5

    So a single scalar separates the two states cleanly:
        diff = ant0 - ant1
        rest    : diff ~ -0.35
        contact : diff  >  ~2 (and always > 1.0 once clearly pressing)

    We use hysteresis so a held contact is one event, not a flicker:
        not-in-contact -> in-contact  when diff > T_ON
        in-contact -> not-in-contact  when diff < T_OFF

This module deliberately does NOT know about the head pose, torque state, the
rhythm, or the state machine above it. It only answers: "are the antennas
touching right now, and did a new touch just start?"
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CollisionConfig:
    # Hysteresis thresholds on diff = ant0 - ant1 (radians).
    # Geometry (from RemiFabre/secret-handshake): antennas splayed at rest give
    # diff ~ -0.35; they meet at the center ("angles equal") at diff ~ 0; firm
    # pressing/flex drives diff to +2..+5.
    #   T_ON  = 0.5 -> just past the touch point, so a deliberate come-together
    #                  registers but rest noise never does.
    #   T_OFF = 0.0 -> antennas must return toward center-or-apart before the
    #                  next touch can count (edge counting for discrete taps).
    # TUNE THESE LIVE with live_contact_probe.py during a real 3-tap gesture.
    t_on: float = 0.5
    t_off: float = 0.0
    # Knock counting (Counter B, see design spec section 5): while contact is
    # held, a "knock" is a local pressure peak whose rise above the running
    # minimum exceeds this prominence (rad), rate-limited by the refractory.
    # ~0.25 rad prominence observed in the recordings; 0.25 default is snug,
    # raise if live testing shows double counts.
    knock_prominence: float = 0.25
    knock_refractory_s: float = 0.15


class CollisionDetector:
    """Stateful edge detector for antenna contact.

    Feed it present antenna positions every tick; it returns whether a NEW
    contact just started (a rising edge). `in_contact` exposes the current
    level for debugging / duration logic.
    """

    def __init__(self, config: CollisionConfig | None = None) -> None:
        self.cfg = config or CollisionConfig()
        self.in_contact: bool = False

    def reset(self) -> None:
        self.in_contact = False

    def update(self, ant0: float, ant1: float) -> bool:
        """Return True exactly on the tick a new contact begins."""
        diff = ant0 - ant1
        if self.in_contact:
            if diff < self.cfg.t_off:
                self.in_contact = False
            return False
        else:
            if diff > self.cfg.t_on:
                self.in_contact = True
                return True
            return False


class KnockDetector:
    """Counts pressure peaks ("knocks") during SUSTAINED contact.

    Counter B from the design spec: the recordings show the natural gesture
    can keep the antennas together and knock without a full release, which
    the edge detector above cannot see. Streaming peak detection: a knock is
    counted when diff rises `knock_prominence` above the running minimum
    since the last counted knock, at most once per refractory period.

    The contact ONSET itself is not reported here (the edge path counts it);
    this only reports the extra knocks inside a held contact.
    """

    def __init__(self, config: CollisionConfig | None = None) -> None:
        self.cfg = config or CollisionConfig()
        self._min_since: float = 0.0
        self._last_knock_t: float = -1e9

    def reset(self) -> None:
        self._min_since = 0.0
        self._last_knock_t = -1e9

    def update(self, t: float, diff: float, in_contact: bool, onset: bool) -> bool:
        """Return True exactly on ticks where a new knock peak is counted."""
        if onset:
            self._min_since = diff
            self._last_knock_t = t
            return False
        if not in_contact:
            return False
        if diff < self._min_since:
            self._min_since = diff
            return False
        if (
            diff - self._min_since >= self.cfg.knock_prominence
            and t - self._last_knock_t >= self.cfg.knock_refractory_s
        ):
            self._min_since = diff
            self._last_knock_t = t
            return True
        return False
