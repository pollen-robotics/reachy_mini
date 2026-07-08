"""Secret handshake: antenna gesture detectors for the control loop.

Two subsystems share one facade, gated by torque state (design spec
docs/superpowers/specs/2026-07-07-antenna-button-handshakes-design.md):

TORQUE OFF -> the WAKE BUTTON
    A freshly built or sleeping Wireless robot is torque off with the head in
    the sleep pose. Knock the antennas together 3 times in quick succession
    (single round) and the caller wakes the robot (`wake_up`: torque on, goto
    base pose, flute "toudoum"). Waking is safe, reversible and obvious, so
    one round of 3 is enough. Armed only while `motor_control_mode ==
    Disabled`, behind the sleep-pose gate, so it is inert during normal use.

TORQUE ON -> the ANTENNA BUTTONS
    While awake (torque on, antennas at the base pose) the four antenna
    directions are soft buttons (antenna_buttons.py) and short coded
    sequences (button_codes.py) trigger actions: WiFi provisioning, torque
    off, an emotion. Armed only while torque is on (the mirror of the
    collision gate). Presses are BASE-RELATIVE: the antennas rest ~10 deg
    outward, so a press is a deviation from that base, not from zero.

This module never touches I/O: `update()` returns an Event and the caller
(RobotBackend._update) decides what each event does.

THE COLLISION LAW (geometric, measured by hand on 3 robots; degrees)

The antennas physically collide "at the center" when BOTH:

    1) l + r  in the working band [-9, 0]
    2) l      in [20, 150]

where l = antenna index 0 (left), r = index 1 (right). A collision EVENT =
entering that region; a firm press flexes through the band and back, so a
0.25 s refractory merges the double crossing into one count.

COST: sub-microsecond per call, no allocation beyond the small press list, no
I/O. Known limitation: `motor_control_mode` does not track per-motor torque
set through `set_motor_torque_ids`, so partially re-enabled motors after a
full disable do not disarm the collision detector. Not reachable in the
first-boot scenario this ships for.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

from .antenna_buttons import AntennaButtonConfig, AntennaButtonDetector
from .button_codes import ButtonCodeConfig, ButtonCodeMachine, CodeEvent

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
# real security is torque-off plus the collision rhythm.
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
    # twice (contact + release re-crossing). A "release latch" was tried and
    # REVERTED: fast tapping has apart windows of only 2-3 ticks, so it
    # dropped knocks. Do not reintroduce without new measured data.


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

    ARMED = "armed"  # collision path armed (no sound recommended)
    WAKE = "wake"  # 3 collisions (torque off) -> wake the robot
    WIFI = "wifi"  # button code (torque on) -> WiFi provisioning
    TORQUE_OFF = "torque_off"  # button code -> disable all torque
    EMOTION = "emotion"  # button code -> play a fixed emotion move
    ORDER_66 = "order_66"  # button code -> play the "order 66" line (gag)


@dataclass(frozen=True)
class HandshakeConfig:
    """All handshake tunables (see the module docstring)."""

    collision: CollisionConfig = field(default_factory=CollisionConfig)
    button: AntennaButtonConfig = field(default_factory=AntennaButtonConfig)
    codes: ButtonCodeConfig = field(default_factory=ButtonCodeConfig)
    taps_required: int = 3  # collisions to wake, single round
    max_gap_s: float = 1.0  # 1 s without a collision -> count resets
    min_tap_spacing_s: float = 0.12  # bounce guard
    arm_settle_s: float = 0.5  # sleep pose held this long -> armed


class _CollisionMachine:
    """idle -> armed -> WAKE (single round of 3 collisions). Pure."""

    def __init__(self, config: HandshakeConfig) -> None:
        self.cfg = config
        self.detector = CollisionDetector(config.collision)
        self.state: str = "idle"
        self._pose_ok_since: float | None = None
        self._taps: list[float] = []

    def reset(self) -> None:
        self.state = "idle"
        self.detector.reset()
        self._pose_ok_since = None
        self._taps.clear()

    def update(self, t: float, ant0: float, ant1: float, pose_ok: bool) -> Event | None:
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

        # state == "armed"
        if onset and self._count_tap(t):
            if len(self._taps) >= self.cfg.taps_required:
                self._rearm()
                return Event.WAKE
        return None

    def _count_tap(self, t: float) -> bool:
        if self._taps and t - self._taps[-1] < self.cfg.min_tap_spacing_s:
            return False
        self._taps.append(t)
        return True

    def _rearm(self) -> None:
        """Return straight to armed after a wake (no idle round-trip).

        Round-tripping through idle forced the pose gate + settle on every
        retry; live, the user's hands jostle the floppy head, the gate
        flickers, and an immediate retry silently fails. Re-entering armed
        directly is safe: at worst the release tail of the final knock counts
        as one stray tap, which expires after max_gap_s.
        """
        self.state = "armed"
        self._pose_ok_since = None
        self._taps.clear()


# Map the pure button-code events to the facade's Event enum.
_CODE_EVENTS = {
    CodeEvent.WIFI: Event.WIFI,
    CodeEvent.TORQUE_OFF: Event.TORQUE_OFF,
    CodeEvent.EMOTION: Event.EMOTION,
    CodeEvent.ORDER_66: Event.ORDER_66,
}


class SecretHandshake:
    """The single entry point: ONE call per control tick.

        event = handshake.update(t, ant0, ant1, head_pose, torque_off)

    t          monotonic seconds (any origin, must only increase)
    ant0/ant1  present antenna joint positions, radians (left, right)
    head_pose  present 4x4 head pose (numpy array or nested sequence)
    torque_off True when motor_control_mode is Disabled (ALL motors off)

    Torque OFF runs the collision -> WAKE path; torque ON runs the
    antenna-button code path. Returns an Event to react to, or None.
    """

    def __init__(self, config: HandshakeConfig | None = None) -> None:
        """Build both detectors; pass a HandshakeConfig to tune them."""
        cfg = config or HandshakeConfig()
        self._collision = _CollisionMachine(cfg)
        self._buttons = AntennaButtonDetector(cfg.button)
        self._codes = ButtonCodeMachine(cfg.codes)

    def update(
        self,
        t: float,
        ant0: float,
        ant1: float,
        head_pose,
        torque_off: bool,
        goal0: float | None = None,
        goal1: float | None = None,
    ) -> Event | None:
        """Feed one control tick; return an Event to react to, or None.

        goal0/goal1 are the commanded antenna targets; button presses are
        deviations from them (robust in any pose, immune to the robot's own
        motion). With no goal the button path stays inert.
        """
        if torque_off:
            # The button path is idle; keep it clean for the next torque-on.
            self._codes.reset()
            self._buttons.reset()
            pose_ok = self._collision.state == "idle" and head_in_sleep_pose(head_pose)
            return self._collision.update(t, ant0, ant1, pose_ok)

        # Torque on: the collision path is inert; run the button codes.
        self._collision.reset()
        presses = self._buttons.update(t, ant0, ant1, goal0, goal1)
        code_event = self._codes.update(t, presses, torque_on=True)
        return _CODE_EVENTS.get(code_event) if code_event is not None else None
