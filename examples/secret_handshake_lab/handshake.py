"""Pure secret-handshake state machine (design spec section 7).

This is the exact logic destined for the 50 Hz control loop, so the rules
from the spec apply hard here:
  - update() is a handful of float compares. No I/O, no meaningful
    allocation, no numpy.
  - It never plays sounds or moves the robot. It RETURNS an event; the caller
    (lab script now, daemon later) decides what a PRIMED beep or an action
    sounds like.
  - Anything partial decays on its own: collisions expire out of a rolling
    window, a primed round times out, torque coming on resets everything.

Two handshakes share the same first round ("the shared prefix"):

    sleep pose + torque off
      -> 3 collisions          => Event.PRIMED      (caller: confirmation beep)
    then, within the timeout:
      -> 3 more collisions     => Event.ACTION_TAPS (v1: the emotion)
      -> hold the antennas gently together ~1 s
                               => Event.ACTION_HOLD (future: WiFi provisioning)

A "collision" is entering the geometric collision region (collision.py,
Remi's measured definition: l+r in a narrow band AND l in [20, 150] deg).
The hold is staying in that region.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from collision import CollisionConfig, CollisionDetector


class Event(enum.Enum):
    ARMED = "armed"  # gate passed, now listening for round 1
    PRIMED = "primed"  # round 1 done -> play the confirmation beep
    ACTION_TAPS = "action_taps"  # handshake A complete (3 taps + 3 taps)
    ACTION_HOLD = "action_hold"  # handshake B complete (3 taps + hold)
    ABORTED = "aborted"  # primed round timed out, back to the start


@dataclass(frozen=True)
class HandshakeConfig:
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    # Round structure.
    taps_required: int = 3
    rhythm_window_s: float = 3.0  # all taps of a round inside this window
    min_tap_spacing_s: float = 0.12  # bounce guard between counted taps
    # Arming gate: head must sit in the sleep pose this long before arming.
    arm_settle_s: float = 0.5
    # Round 2.
    primed_refractory_s: float = 0.25  # ignore contact right after the beep
    primed_timeout_s: float = 8.0  # no valid round 2 -> abort
    hold_min_s: float = 1.0  # stay in the collision region for handshake B
    hold_grace_s: float = 0.3  # membership may flicker briefly during a hold
    # After an action fires, stay inert this long before re-arming.
    cooldown_s: float = 2.0


class HandshakeStateMachine:
    """Feed it one sample per control tick; it returns an Event or None.

    update(t, ant0, ant1, torque_off, head_in_sleep_pose)
      t                  monotonic seconds (any origin, must only increase)
      ant0, ant1         present antenna joint positions, radians (left, right)
      torque_off         True when motor_control_mode is Disabled
      head_in_sleep_pose result of the pose gate (pose_gate.py); only sampled
                         while idle, so jostling the head mid-gesture is fine
    """

    def __init__(self, config: HandshakeConfig | None = None) -> None:
        self.cfg = config or HandshakeConfig()
        self.detector = CollisionDetector(self.cfg.collision)
        self.state: str = "idle"
        self._pose_ok_since: float | None = None
        self._taps: list[float] = []
        self._primed_at: float = 0.0
        self._hold_since: float | None = None  # in the region (while primed)
        self._last_in_region_t: float = 0.0
        self._cooldown_until: float = 0.0

    @property
    def tap_count(self) -> int:
        """Collisions currently inside the rhythm window (for live display)."""
        return len(self._taps)

    def reset(self) -> None:
        self.state = "idle"
        self.detector.reset()
        self._pose_ok_since = None
        self._taps.clear()
        self._hold_since = None

    def update(
        self,
        t: float,
        ant0: float,
        ant1: float,
        torque_off: bool,
        head_in_sleep_pose: bool,
    ) -> Event | None:
        if not torque_off:
            if self.state != "idle" or self._pose_ok_since is not None:
                self.reset()
            return None

        # Let stale taps fall out of the rolling window (one per tick is
        # enough at 50 Hz, and keeps the per-tick cost constant).
        if self._taps and t - self._taps[0] > self.cfg.rhythm_window_s:
            self._taps.pop(0)

        onset = self.detector.update(t, ant0, ant1)

        if self.state == "idle":
            if head_in_sleep_pose and t >= self._cooldown_until:
                if self._pose_ok_since is None:
                    self._pose_ok_since = t
                elif t - self._pose_ok_since >= self.cfg.arm_settle_s:
                    self.state = "armed"
                    self._taps.clear()
                    return Event.ARMED
            else:
                self._pose_ok_since = None
            return None

        if self.state == "armed":
            if onset and self._count_tap(t):
                if len(self._taps) >= self.cfg.taps_required:
                    self.state = "primed"
                    self._primed_at = t
                    self._taps.clear()
                    self._hold_since = None
                    return Event.PRIMED
            return None

        # state == "primed"
        if t - self._primed_at > self.cfg.primed_timeout_s:
            self._back_to_idle(t, cooldown_s=0.0)
            return Event.ABORTED
        if t - self._primed_at < self.cfg.primed_refractory_s:
            return None
        if onset:
            if self._count_tap(t) and len(self._taps) >= self.cfg.taps_required:
                self._back_to_idle(t, cooldown_s=self.cfg.cooldown_s)
                return Event.ACTION_TAPS
        if self.detector.in_collision:
            self._last_in_region_t = t
            if self._hold_since is None:
                self._hold_since = t
            elif t - self._hold_since >= self.cfg.hold_min_s:
                self._back_to_idle(t, cooldown_s=self.cfg.cooldown_s)
                return Event.ACTION_HOLD
        elif (
            self._hold_since is not None
            and t - self._last_in_region_t > self.cfg.hold_grace_s
        ):
            self._hold_since = None
        return None

    def _count_tap(self, t: float) -> bool:
        """Register a collision as a tap. False if it is a bounce."""
        if self._taps and t - self._taps[-1] < self.cfg.min_tap_spacing_s:
            return False
        self._taps.append(t)
        while self._taps and t - self._taps[0] > self.cfg.rhythm_window_s:
            self._taps.pop(0)
        return True

    def _back_to_idle(self, t: float, cooldown_s: float) -> None:
        self.state = "idle"
        self._pose_ok_since = None
        self._taps.clear()
        self._hold_since = None
        self._cooldown_until = t + cooldown_s
