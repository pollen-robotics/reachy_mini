"""Secret handshake: antenna-collision gesture detector for the control loop.

Right after a Wireless Reachy Mini is built it has no WiFi and cannot run
any app. This detector gives it a first sign of life that works fully
offline: with all motors torque OFF and the head resting in the sleep pose,
knock the antennas together 3 times (confirmation beep), then 3 more times
within a few seconds (success sound). The caller decides what each returned
Event does; this module never touches I/O.

THE GESTURE AND ITS SAFETY

Armed ONLY while `motor_control_mode == Disabled` (all motors torque off),
so it is inert during any normal robot use: apps, moves, teleoperation all
run with torque on. A freshly booted Wireless robot is torque off, so it is
armed out of the box. The two-round rhythm plus the sleep-pose gate make an
accidental trigger while manipulating a floppy robot very unlikely.

THE COLLISION LAW (geometric, measured by hand on 3 robots; degrees)

The antennas physically collide "at the center" when BOTH:

    1) l + r  in the working band [-9, 0]   (measured [-7, -3] + margin,
                                             top relaxed to 0 = the
                                             theoretical symmetric touch)
    2) l      in [20, 150]

where l = antenna index 0 (left), r = index 1 (right), as returned by
`get_present_antenna_joint_positions()`. A collision EVENT = entering that
region; a firm press flexes the antennas through the band and back, so a
0.25 s refractory merges the double crossing into one count. Validated on
the RemiFabre/secret-handshake recordings: exactly 3 collisions found on
each recorded gesture, zero on 30 s of adversarial slide data, rest and
single-antenna play inert (full evidence: examples/secret_handshake_lab/).

=============================================================================
TUNABLES AT A GLANCE (single source of truth: the two dataclasses below)

  collision region      l + r in [-9, 0] deg  AND  l in [20, 150] deg
  collision debounce    0.25 s refractory (a >0.25 s press counts twice:
                        known accepted quirk, see CollisionConfig)
  collisions per round  3, in quick succession
  sequence reset        1.0 s without a collision -> count goes back to 0
  arming settle         head in sleep pose for 0.5 s -> armed (idle only:
                        boot and torque-off transitions)
  round 2 window        3.0 s after the prime beep -> aborted
  after success/abort   straight back to armed (immediate retry works)
=============================================================================

COST: sub-microsecond per call (see examples/secret_handshake_lab/bench.py:
115 ns on the torque-ON path that runs during all normal use, <1 us worst
case under continuous tapping, on an M-series Mac). No allocation, no I/O.

Known limitation: `motor_control_mode` does not track per-motor torque set
through `set_motor_torque_ids`, so partially re-enabled motors after a full
disable do not disarm the detector. Not reachable in the first-boot
scenario this ships for.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

_RAD2DEG = 180.0 / math.pi


def _wrap_deg(deg: float) -> float:
    """Wrap a multi-turn angle to [-180, 180).

    The antenna encoders are multi-turn: handling the floppy antennas can
    park them whole turns away from the calibrated zero (seen live at rest:
    l=-340, r=+334, physically l=+20, r=-26). The law is about the physical
    angle, so it must apply to the wrapped reading.
    """
    return (deg + 180.0) % 360.0 - 180.0

# Copy of ReachyMiniBackend.SLEEP_HEAD_POSE (abstract.py), kept local so this
# module stays import-free and trivially testable.
_SLEEP_HEAD_POSE = (
    (0.911, 0.004, 0.413, -0.021),
    (-0.004, 1.0, -0.001, 0.001),
    (-0.413, -0.001, 0.911, -0.044),
    (0.0, 0.0, 0.0, 1.0),
)

# Sleep-pose gate tolerances, generous on purpose: it is a sanity gate, the
# real security is torque-off plus the two-round rhythm. Derived from the
# recorded gestures (head settled within these margins in all recordings).
_TOL_X_M = 0.015
_TOL_Y_M = 0.012
_TOL_Z_M = 0.012
_TOL_ROLL_RAD = math.radians(8.0)
_TOL_PITCH_RAD = math.radians(15.0)


def _pitch_roll(pose) -> tuple[float, float]:
    s = -pose[2][0]
    s = max(-1.0, min(1.0, s))
    return math.asin(s), math.atan2(pose[2][1], pose[2][2])


_REF_PITCH, _REF_ROLL = _pitch_roll(_SLEEP_HEAD_POSE)


def head_in_sleep_pose(pose) -> bool:
    """Return True when the 4x4 pose is roughly the sleep pose (yaw ignored)."""
    if abs(pose[0][3] - _SLEEP_HEAD_POSE[0][3]) > _TOL_X_M:
        return False
    if abs(pose[1][3] - _SLEEP_HEAD_POSE[1][3]) > _TOL_Y_M:
        return False
    if abs(pose[2][3] - _SLEEP_HEAD_POSE[2][3]) > _TOL_Z_M:
        return False
    pitch, roll = _pitch_roll(pose)
    if abs(pitch - _REF_PITCH) > _TOL_PITCH_RAD:
        return False
    if abs(roll - _REF_ROLL) > _TOL_ROLL_RAD:
        return False
    return True


@dataclass(frozen=True)
class CollisionConfig:
    """The geometric collision region, in degrees (see module docstring)."""

    sum_min_deg: float = -7.0  # measured band on 2 robots
    sum_max_deg: float = -2.0  # relaxed so the margined top sits at 0
    l_min_deg: float = 20.0
    l_max_deg: float = 150.0
    margin_deg: float = 4.0  # widens the sum band by margin/2 on each side
    refractory_s: float = 0.25  # one count per knock (press = double crossing)
    # KNOWN QUIRK, accepted: a press held longer than the refractory counts
    # twice (contact + release re-crossing), so a prime can happen in 2 firm
    # knocks. A "release latch" (require a not-pressed dwell between counts)
    # was tried and REVERTED: live fast tapping has apart windows of only
    # 2-3 ticks, under any dwell threshold that still blocks the release
    # crossing, so fast gestures dropped knocks (far worse than an
    # occasional extra count). Velocity-based discrimination is rejected
    # by design; do not reintroduce either without new measured data.


class CollisionDetector:
    """Edge detector for the collision region. update() -> True per collision."""

    def __init__(self, config: CollisionConfig | None = None) -> None:
        """Precompute the margined band bounds from the config."""
        self.cfg = config or CollisionConfig()
        half = self.cfg.margin_deg / 2.0
        self.sum_lo_deg = self.cfg.sum_min_deg - half
        self.sum_hi_deg = self.cfg.sum_max_deg + half
        self.in_collision: bool = False
        self._last_onset_t: float = -1e9

    def reset(self) -> None:
        """Forget any ongoing collision and refractory state."""
        self.in_collision = False
        self._last_onset_t = -1e9

    def update(self, t: float, ant0: float, ant1: float) -> bool:
        """Return True exactly on the tick a new collision is seen."""
        l_deg = _wrap_deg(ant0 * _RAD2DEG)
        # Wrap the sum too: l and r wrapped independently can land one full
        # turn apart (e.g. l=+170, r=-190 -> wrapped r=+170, sum=+340).
        sum_deg = _wrap_deg(l_deg + _wrap_deg(ant1 * _RAD2DEG))
        inside = (
            self.cfg.l_min_deg <= l_deg <= self.cfg.l_max_deg
            and self.sum_lo_deg <= sum_deg <= self.sum_hi_deg
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


class Event(enum.Enum):
    """What just happened; the caller decides what each event does."""

    ARMED = "armed"  # gate passed, listening (no sound recommended)
    PRIMED = "primed"  # first 3 collisions -> play the confirmation sound
    SUCCESS = "success"  # 3 more collisions -> play the success sound
    ABORTED = "aborted"  # round 2 did not happen in time


@dataclass(frozen=True)
class HandshakeConfig:
    """All handshake tunables (see the banner in the module docstring)."""

    collision: CollisionConfig = field(default_factory=CollisionConfig)
    taps_required: int = 3
    max_gap_s: float = 1.0  # 1 s without a collision -> sequence resets
    min_tap_spacing_s: float = 0.12  # bounce guard
    arm_settle_s: float = 0.5  # sleep pose held this long -> armed
    primed_refractory_s: float = 0.25  # ignore contact right after the beep
    primed_timeout_s: float = 3.0  # time allowed for round 2


class _StateMachine:
    """idle -> armed -> primed -> (success | aborted). Pure, event-driven."""

    def __init__(self, config: HandshakeConfig) -> None:
        self.cfg = config
        self.detector = CollisionDetector(config.collision)
        self.state: str = "idle"
        self._pose_ok_since: float | None = None
        self._taps: list[float] = []
        self._primed_at: float = 0.0

    def reset(self) -> None:
        self.state = "idle"
        self.detector.reset()
        self._pose_ok_since = None
        self._taps.clear()

    def update(
        self, t: float, ant0: float, ant1: float, torque_off: bool, pose_ok: bool
    ) -> Event | None:
        if not torque_off:
            if self.state != "idle" or self._pose_ok_since is not None:
                self.reset()
            return None

        if self._taps and t - self._taps[-1] > self.cfg.max_gap_s:
            self._taps.clear()

        onset = self.detector.update(t, ant0, ant1)

        if self.state == "idle":
            if pose_ok:
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
                    return Event.PRIMED
            return None

        # state == "primed"
        if t - self._primed_at > self.cfg.primed_timeout_s:
            self._rearm()
            return Event.ABORTED
        if t - self._primed_at < self.cfg.primed_refractory_s:
            return None
        if onset and self._count_tap(t):
            if len(self._taps) >= self.cfg.taps_required:
                self._rearm()
                return Event.SUCCESS
        return None

    def _count_tap(self, t: float) -> bool:
        if self._taps and t - self._taps[-1] < self.cfg.min_tap_spacing_s:
            return False
        self._taps.append(t)
        return True

    def _rearm(self) -> None:
        """Return straight to armed after a success or an abort.

        Round-tripping through idle forced the pose gate + settle (plus a
        cooldown) on every retry; live, the user's hands jostle the floppy
        head, the gate flickers, and an immediate retry silently fails.
        Re-entering armed directly is safe: at worst the release tail of
        the final knock counts as one stray tap, which expires after
        max_gap_s without ever reaching the 3 needed to prime.
        """
        self.state = "armed"
        self._pose_ok_since = None
        self._taps.clear()


class SecretHandshake:
    """The single entry point: ONE call per control tick.

        event = handshake.update(t, ant0, ant1, head_pose, torque_off)

    t          monotonic seconds (any origin, must only increase)
    ant0/ant1  present antenna joint positions, radians (left, right)
    head_pose  present 4x4 head pose (numpy array or nested sequence)
    torque_off True when motor_control_mode is Disabled (ALL motors off)

    Returns an Event to react to (play a sound), or None on most ticks.
    """

    def __init__(self, config: HandshakeConfig | None = None) -> None:
        """Build the detector; pass a HandshakeConfig to tune it."""
        self.machine = _StateMachine(config or HandshakeConfig())

    def update(
        self, t: float, ant0: float, ant1: float, head_pose, torque_off: bool
    ) -> Event | None:
        """Feed one control tick; return an Event to react to, or None."""
        # The machine only consults the pose gate while idle: skip the 4x4
        # checks everywhere else, including the torque-ON path that runs
        # during all normal robot use.
        pose_ok = (
            torque_off
            and self.machine.state == "idle"
            and head_in_sleep_pose(head_pose)
        )
        return self.machine.update(t, ant0, ant1, torque_off, pose_ok)
