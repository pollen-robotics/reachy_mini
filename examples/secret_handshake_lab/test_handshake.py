"""Unit tests for the geometric collision law and the handshake machine.

Everything here is synthetic and offline: we generate 50 Hz antenna samples
that mimic the real gesture (checked against the HF recordings, see
replay_validate.py) and feed them tick by tick, exactly like the daemon
control loop will.

The collision definition (measured by Remi on 2 robots, in degrees):
    1) l_ant + r_ant in [-7, -3]   (slight asymmetry: not exactly 0)
    2) l_ant in [20, 150]
widened by a margin (default 4 deg -> sum band becomes [-9, -1]).
A collision event = entering that region, with a refractory that merges the
double band-crossing of a firm press (in through the band, out beyond it
while flexing, back through on release).

Run:
    python -m pytest examples/secret_handshake_lab/test_handshake.py -v
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collision import CollisionConfig, CollisionDetector  # noqa: E402
from handshake import (  # noqa: E402
    Event,
    HandshakeConfig,
    HandshakeStateMachine,
    SecretHandshake,
)
from pose_gate import SLEEP_HEAD_POSE, head_in_sleep_pose  # noqa: E402

DT = 0.02  # 50 Hz, same as the daemon control loop


def d2r(deg: float) -> float:
    return math.radians(deg)


# Characteristic antenna configurations, in degrees (l, r):
REST = (-12.0, 10.0)  # floppy rest: sum=-2 is in band, but l fails cond 2
CONTACT = (60.0, -65.0)  # touching at center: sum=-5, l in range
PRESS = (65.0, -30.0)  # firm press, antennas flexed: sum=+35, out of band
CROSSED = (149.0, -160.0)  # slid past each other and parked: sum=-11


def seg(sample: tuple[float, float], seconds: float) -> list[tuple[float, float]]:
    return [(d2r(sample[0]), d2r(sample[1]))] * max(1, round(seconds / DT))


def tap(release_s: float = 0.3) -> list[tuple[float, float]]:
    """One collision: pass into the band, press through it, release."""
    return (
        seg(CONTACT, 0.04)
        + seg(PRESS, 0.06)
        + seg(CONTACT, 0.04)  # back through the band on the way out
        + seg(REST, release_s)
    )


def taps(n: int, release_s: float = 0.3) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for _ in range(n):
        out += tap(release_s)
    return out


# ---------------------------------------------------------------------------
# Collision detector: the geometric law
# ---------------------------------------------------------------------------


def run_detector(
    samples: list[tuple[float, float]], config: CollisionConfig | None = None
) -> tuple[int, int]:
    det = CollisionDetector(config or CollisionConfig())
    onsets = 0
    in_ticks = 0
    t = 0.0
    for a0, a1 in samples:
        if det.update(t, a0, a1):
            onsets += 1
        if det.in_collision:
            in_ticks += 1
        t += DT
    return onsets, in_ticks


def test_detector_counts_one_collision_per_tap() -> None:
    onsets, _ = run_detector(seg(REST, 0.5) + taps(3))
    assert onsets == 3


def test_detector_press_through_band_counts_once() -> None:
    # One tap enters the band twice (in, then back out through it); the
    # refractory must merge that into a single collision.
    onsets, _ = run_detector(seg(REST, 0.5) + tap())
    assert onsets == 1


def test_detector_rest_is_not_a_collision() -> None:
    # At rest sum=-2 sits in the band, but l=-12 fails condition 2.
    onsets, in_ticks = run_detector(seg(REST, 2.0))
    assert onsets == 0
    assert in_ticks == 0


def test_detector_press_alone_is_not_in_band() -> None:
    onsets, in_ticks = run_detector(seg(PRESS, 1.0))
    assert in_ticks == 0
    assert onsets == 0


def test_detector_crossed_park_is_not_a_collision() -> None:
    # Antennas slid past each other and parked: sum=-11, below the band.
    onsets, in_ticks = run_detector(seg(CROSSED, 2.0))
    assert onsets == 0
    assert in_ticks == 0


def test_detector_single_antenna_swing_is_inert() -> None:
    # Swing only the left antenna across its whole range, right stays at
    # rest (+10): sum = l+10 is in [-9,-1] only while l is in [-19,-11],
    # where condition 2 (l >= 20) fails. No collision.
    samples = [(d2r(l), d2r(10.0)) for l in range(-170, 171, 2)]
    onsets, in_ticks = run_detector(samples)
    assert onsets == 0
    assert in_ticks == 0


def test_detector_margin_widens_the_band() -> None:
    just_outside = (60.0, -68.5)  # sum=-8.5: outside [-7,-3], inside [-9,-1]
    strict = CollisionConfig(margin_deg=0.0)
    onsets, in_ticks = run_detector(seg(REST, 0.3) + seg(just_outside, 0.5), strict)
    assert in_ticks == 0 and onsets == 0
    # default margin 4 (REST first: the release latch needs a separation)
    onsets, in_ticks = run_detector(seg(REST, 0.3) + seg(just_outside, 0.5))
    assert in_ticks > 0 and onsets == 1


def test_detector_hold_is_one_collision_and_stays_in_band() -> None:
    onsets, in_ticks = run_detector(seg(REST, 0.5) + seg(CONTACT, 1.5))
    assert onsets == 1
    assert in_ticks >= 70  # ~1.5 s at 50 Hz


def test_detector_full_turn_offsets_are_normalized() -> None:
    """Taps must be detected even when the encoders carry whole turns."""
    # The antenna encoders are multi-turn: handling the floppy antennas can
    # park them a whole turn away from where they were calibrated (seen live
    # on a wireless robot at rest: l=-340 deg, r=+334 deg, physically
    # l=+20, r=-26). The law must apply to the wrapped angle, not the raw
    # multi-turn reading.
    turn = 2 * math.pi
    for l_turns, r_turns in [(-1, 1), (1, -1), (2, 0), (0, -2)]:
        samples = [
            (a0 + l_turns * turn, a1 + r_turns * turn)
            for a0, a1 in seg(REST, 0.5) + taps(3)
        ]
        onsets, _ = run_detector(samples)
        assert onsets == 3, (l_turns, r_turns)


def test_detector_crossed_park_with_turn_offset_is_inert() -> None:
    """Wrapping must not turn the parked-crossed state into a collision."""
    turn = 2 * math.pi
    samples = [(a0 - turn, a1 + turn) for a0, a1 in seg(CROSSED, 2.0)]
    onsets, in_ticks = run_detector(samples)
    assert onsets == 0
    assert in_ticks == 0


def test_detector_slow_press_counts_once() -> None:
    """One knock is one collision even when the press outlasts the refractory."""
    # A firm knock passes through the band twice (contact, then again while
    # releasing). When the press lasts longer than the 0.25 s refractory the
    # release crossing used to count as a second collision (seen live:
    # primed after 2 knocks). Counting must require a not-pressed dwell in
    # between (the release latch), not just elapsed time.
    slow_knock = seg(CONTACT, 0.04) + seg(PRESS, 0.4) + seg(CONTACT, 0.04)
    onsets, _ = run_detector(seg(REST, 0.5) + slow_knock + seg(REST, 0.5))
    assert onsets == 1


def test_detector_starting_inside_region_is_inert_until_released() -> None:
    """Antennas resting crossed at the touch point must not pre-count."""
    # Seen live: the antennas can be parked touching at the center. Sitting
    # there (or noise-jittering around the region edge) must count nothing;
    # only after a real separation may collisions count again.
    onsets, _ = run_detector(seg(CONTACT, 2.0))
    assert onsets == 0
    onsets, _ = run_detector(seg(CONTACT, 2.0) + seg(REST, 0.3) + tap())
    assert onsets == 1


# ---------------------------------------------------------------------------
# State machine helpers
# ---------------------------------------------------------------------------


class Runner:
    def __init__(self, machine: HandshakeStateMachine) -> None:
        self.machine = machine
        self.t = 0.0
        self.events: list[tuple[float, Event]] = []

    def feed(
        self,
        samples: list[tuple[float, float]],
        torque_off: bool = True,
        pose_ok: bool = True,
    ) -> None:
        for ant0, ant1 in samples:
            e = self.machine.update(
                self.t, ant0, ant1, torque_off=torque_off, head_in_sleep_pose=pose_ok
            )
            if e is not None:
                self.events.append((round(self.t, 2), e))
            self.t += DT

    def event_kinds(self) -> list[Event]:
        return [e for _, e in self.events]


def make_runner() -> Runner:
    return Runner(HandshakeStateMachine(HandshakeConfig()))


# ---------------------------------------------------------------------------
# Arming gate
# ---------------------------------------------------------------------------


def test_arms_after_settling_in_sleep_pose() -> None:
    r = make_runner()
    r.feed(seg(REST, 1.0))
    assert r.event_kinds() == [Event.ARMED]
    assert r.machine.state == "armed"


def test_does_not_arm_without_sleep_pose() -> None:
    r = make_runner()
    r.feed(seg(REST, 2.0), pose_ok=False)
    assert r.event_kinds() == []
    assert r.machine.state == "idle"


def test_does_not_arm_with_torque_on() -> None:
    r = make_runner()
    r.feed(seg(REST, 2.0), torque_off=False)
    assert r.event_kinds() == []
    assert r.machine.state == "idle"


def test_pose_flicker_restarts_settle_timer() -> None:
    r = make_runner()
    r.feed(seg(REST, 0.3))
    r.feed(seg(REST, 0.1), pose_ok=False)
    r.feed(seg(REST, 0.3))
    assert r.event_kinds() == []  # 0.3 s < settle time, must not be armed yet
    r.feed(seg(REST, 0.4))
    assert r.event_kinds() == [Event.ARMED]


# ---------------------------------------------------------------------------
# Handshake A: 3 collisions -> beep -> 3 collisions -> action
# ---------------------------------------------------------------------------


def test_full_tap_tap_handshake() -> None:
    r = make_runner()
    r.feed(seg(REST, 1.0) + taps(3))
    assert r.event_kinds() == [Event.ARMED, Event.PRIMED]
    r.feed(seg(REST, 0.5) + taps(3))
    assert r.event_kinds() == [Event.ARMED, Event.PRIMED, Event.ACTION_TAPS]
    assert r.machine.state == "armed"  # actions re-arm directly


def test_two_taps_do_not_prime() -> None:
    r = make_runner()
    r.feed(seg(REST, 1.0) + taps(2))
    assert r.machine.tap_count == 2  # live UI reads this
    r.feed(seg(REST, 1.5))  # 1 s without a collision resets the sequence
    assert Event.PRIMED not in r.event_kinds()
    assert r.machine.tap_count == 0


def test_slow_taps_never_prime() -> None:
    r = make_runner()
    r.feed(seg(REST, 1.0) + taps(4, release_s=1.2))  # gaps > 1 s: sequence dies
    assert Event.PRIMED not in r.event_kinds()


def test_one_second_gap_restarts_the_count() -> None:
    r = make_runner()
    r.feed(seg(REST, 1.0) + taps(2))
    r.feed(seg(REST, 1.2))  # sequence reset
    r.feed(tap())
    assert r.machine.tap_count == 1  # this tap starts a NEW sequence
    r.feed(taps(2))
    assert r.event_kinds()[-1] == Event.PRIMED  # 3 quick ones prime


def test_single_antenna_play_is_inert() -> None:
    r = make_runner()
    r.feed(seg(REST, 1.0))
    swing = [(d2r(l), d2r(10.0)) for l in range(-170, 171, 2)]
    r.feed(swing + swing)
    assert r.event_kinds() == [Event.ARMED]
    assert r.machine.tap_count == 0


# ---------------------------------------------------------------------------
# Handshake B: 3 collisions -> beep -> gentle hold in the band -> action
# ---------------------------------------------------------------------------


def test_hold_handshake() -> None:
    r = make_runner()
    r.feed(seg(REST, 1.0) + taps(3))
    r.feed(seg(REST, 0.5) + seg(CONTACT, 1.6))
    assert r.event_kinds() == [Event.ARMED, Event.PRIMED, Event.ACTION_HOLD]
    assert r.machine.state == "armed"  # actions re-arm directly


def test_hold_in_round_one_does_not_prime() -> None:
    r = make_runner()
    r.feed(seg(REST, 1.0) + seg(CONTACT, 2.0) + seg(REST, 0.5))
    assert Event.PRIMED not in r.event_kinds()  # a hold is 1 collision, not 3


def test_taps_do_not_add_up_to_a_hold() -> None:
    # Two taps then waiting must not fire ACTION_HOLD: each tap leaves the
    # band quickly, so the hold timer never accumulates.
    r = make_runner()
    r.feed(seg(REST, 1.0) + taps(3))
    r.feed(seg(REST, 0.5) + taps(2) + seg(REST, 1.5))
    assert Event.ACTION_HOLD not in r.event_kinds()


# ---------------------------------------------------------------------------
# Timeouts, resets, robustness
# ---------------------------------------------------------------------------


def test_primed_times_out_after_3s_then_recovers() -> None:
    r = make_runner()
    r.feed(seg(REST, 1.0) + taps(3))
    r.feed(seg(REST, 2.5))  # not timed out yet (3 s round-2 window)
    assert r.event_kinds()[-1] == Event.PRIMED
    r.feed(seg(REST, 1.0))  # now it is
    kinds = r.event_kinds()
    assert kinds[:3] == [Event.ARMED, Event.PRIMED, Event.ABORTED]
    r.feed(taps(3))
    assert r.event_kinds()[-1] == Event.PRIMED


def test_torque_on_resets_everything() -> None:
    r = make_runner()
    r.feed(seg(REST, 1.0) + taps(2))
    r.feed(tap(), torque_off=False)  # torque comes on during collision 3
    assert r.machine.state == "idle"
    n_before = len(r.events)
    r.feed(taps(3), torque_off=False)  # taps while torque on: inert
    assert len(r.events) == n_before


def test_handshake_is_repeatable() -> None:
    r = make_runner()
    for _ in range(2):
        r.feed(seg(REST, 3.5))  # a calm pause between rounds still works
        r.feed(taps(3))
        r.feed(seg(REST, 0.5) + taps(3))
    assert r.event_kinds().count(Event.ACTION_TAPS) == 2


def test_immediate_retry_after_action() -> None:
    """After an action fires the machine is instantly ready for a new round."""
    # Live report: retrying right after a success (or abort) silently failed
    # because the machine dropped to idle and had to re-pass the pose gate
    # (0.5 s settle) plus a 2 s cooldown, while the user's hands jostle the
    # floppy head. Success/abort must return the machine straight to armed.
    r = make_runner()
    r.feed(seg(REST, 1.0) + taps(3))
    r.feed(seg(REST, 0.5) + taps(3))
    assert r.event_kinds()[-1] == Event.ACTION_TAPS
    r.feed(taps(3))  # retry immediately, no pause
    assert r.event_kinds()[-1] == Event.PRIMED
    r.feed(seg(REST, 0.5) + taps(3))
    assert r.event_kinds()[-1] == Event.ACTION_TAPS


def test_immediate_retry_after_abort() -> None:
    """A failed round 2 must be retryable immediately, no settle wait."""
    r = make_runner()
    r.feed(seg(REST, 1.0) + taps(3))
    r.feed(seg(REST, 3.2))  # blow the round-2 window
    assert r.event_kinds()[-1] == Event.ABORTED
    r.feed(taps(3))  # retry immediately
    assert r.event_kinds()[-1] == Event.PRIMED


def test_holding_through_an_action_does_not_refire() -> None:
    """Keeping the antennas together after a hold action stays inert."""
    r = make_runner()
    r.feed(seg(REST, 1.0) + taps(3))
    r.feed(seg(REST, 0.5) + seg(CONTACT, 1.2))
    assert r.event_kinds()[-1] == Event.ACTION_HOLD
    r.feed(seg(CONTACT, 3.0))  # still holding: nothing may fire
    assert r.event_kinds()[-1] == Event.ACTION_HOLD
    r.feed(seg(REST, 0.5) + taps(3))  # release, then a fresh round works
    assert r.event_kinds()[-1] == Event.PRIMED


# ---------------------------------------------------------------------------
# SecretHandshake: the single function the daemon control loop will call
# ---------------------------------------------------------------------------


def test_facade_runs_the_full_handshake_from_raw_inputs() -> None:
    hs = SecretHandshake()
    t = 0.0
    events = []
    for ant0, ant1 in seg(REST, 1.0) + taps(3) + seg(REST, 0.5) + taps(3):
        e = hs.update(t, ant0, ant1, head_pose=SLEEP_HEAD_POSE, torque_off=True)
        if e is not None:
            events.append(e)
        t += DT
    assert events == [Event.ARMED, Event.PRIMED, Event.ACTION_TAPS]


def test_facade_pose_gate_blocks_arming() -> None:
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    hs = SecretHandshake()
    t = 0.0
    for ant0, ant1 in seg(REST, 2.0):
        assert hs.update(t, ant0, ant1, head_pose=identity, torque_off=True) is None
        t += DT


# ---------------------------------------------------------------------------
# Pose gate helper
# ---------------------------------------------------------------------------


def test_sleep_pose_passes_gate() -> None:
    assert head_in_sleep_pose(SLEEP_HEAD_POSE)


def test_wake_pose_fails_gate() -> None:
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    assert not head_in_sleep_pose(identity)


def test_small_deviation_passes_gate() -> None:
    pose = [list(row) for row in SLEEP_HEAD_POSE]
    pose[2][3] += 0.008  # z off by 8 mm, within the +-12 mm tolerance
    assert head_in_sleep_pose(pose)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
