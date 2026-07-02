"""Unit tests for the pure HandshakeStateMachine (and the pose gate helper).

Everything here is synthetic and offline: we generate 50 Hz antenna samples
that mimic the real gesture timing and feed them tick by tick, exactly like
the daemon control loop will. No hardware, no network.

Run:
    python -m pytest examples/secret_handshake_lab/test_handshake.py -v
or directly:
    python examples/secret_handshake_lab/test_handshake.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from handshake import Event, HandshakeConfig, HandshakeStateMachine  # noqa: E402
from pose_gate import SLEEP_HEAD_POSE, head_in_sleep_pose  # noqa: E402

DT = 0.02  # 50 Hz, same as the daemon control loop

# Antenna positions (radians). From the recorded data: rest is splayed
# (diff ~ -0.35), a deliberate press reaches diff ~ +2.5.
REST = (-0.175, 0.175)
TOUCH = (1.25, -1.25)


def seg(sample: tuple[float, float], seconds: float) -> list[tuple[float, float]]:
    """A constant-antenna segment of the given duration, sampled at 50 Hz."""
    return [sample] * max(1, round(seconds / DT))


def tap(release_s: float = 0.3) -> list[tuple[float, float]]:
    """One quick tap: short contact then release."""
    return seg(TOUCH, 0.10) + seg(REST, release_s)


def three_taps() -> list[tuple[float, float]]:
    return tap() + tap() + tap()


class Runner:
    """Feeds samples to a machine with a monotonic clock, collects events."""

    def __init__(self, machine: HandshakeStateMachine) -> None:
        self.machine = machine
        self.t = 0.0
        self.events: list[tuple[float, Event]] = []

    def feed(
        self,
        samples: list[tuple[float, float]],
        torque_off: bool = True,
        pose_ok: bool = True,
    ) -> list[Event]:
        new: list[Event] = []
        for ant0, ant1 in samples:
            e = self.machine.update(
                self.t, ant0, ant1, torque_off=torque_off, head_in_sleep_pose=pose_ok
            )
            if e is not None:
                self.events.append((round(self.t, 2), e))
                new.append(e)
            self.t += DT
        return new

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
    r.feed(seg(REST, 0.1), pose_ok=False)  # pose blip before settle completes
    r.feed(seg(REST, 0.3))
    assert r.event_kinds() == []  # 0.3 s < settle time, must not be armed yet
    r.feed(seg(REST, 0.4))
    assert r.event_kinds() == [Event.ARMED]


# ---------------------------------------------------------------------------
# Handshake A: 3 taps -> beep -> 3 taps -> action
# ---------------------------------------------------------------------------


def test_full_tap_tap_handshake() -> None:
    r = make_runner()
    r.feed(seg(REST, 1.0))
    r.feed(three_taps())
    assert r.event_kinds() == [Event.ARMED, Event.PRIMED]
    r.feed(seg(REST, 0.5))
    r.feed(three_taps())
    assert r.event_kinds() == [Event.ARMED, Event.PRIMED, Event.ACTION_TAPS]
    assert r.machine.state == "idle"  # back to start after the action


def test_two_taps_do_not_prime() -> None:
    r = make_runner()
    r.feed(seg(REST, 1.0))
    r.feed(tap() + tap())
    assert r.machine.tap_count == 2  # live UI reads this
    r.feed(seg(REST, 4.0))  # let the rhythm window expire
    assert Event.PRIMED not in r.event_kinds()
    assert r.machine.tap_count == 0


def test_slow_taps_never_prime() -> None:
    r = make_runner()
    r.feed(seg(REST, 1.0))
    for _ in range(4):
        r.feed(tap(release_s=1.7))  # 3 s rolling window can only ever hold 2
    assert Event.PRIMED not in r.event_kinds()


# ---------------------------------------------------------------------------
# Handshake B: 3 taps -> beep -> long hold -> other action
# ---------------------------------------------------------------------------


def test_hold_handshake() -> None:
    r = make_runner()
    r.feed(seg(REST, 1.0))
    r.feed(three_taps())
    r.feed(seg(REST, 0.5))
    r.feed(seg(TOUCH, 1.6))  # press and hold
    assert r.event_kinds() == [Event.ARMED, Event.PRIMED, Event.ACTION_HOLD]
    assert r.machine.state == "idle"


def test_contact_held_across_prime_does_not_trigger_hold() -> None:
    # Third round-1 "tap" is pressed and never released: it primes, but the
    # ongoing contact started before PRIMED so it must not count as the hold.
    r = make_runner()
    r.feed(seg(REST, 1.0))
    r.feed(tap() + tap())
    r.feed(seg(TOUCH, 3.0))  # press for the 3rd tap and keep holding
    assert r.event_kinds() == [Event.ARMED, Event.PRIMED]
    assert Event.ACTION_HOLD not in r.event_kinds()


# ---------------------------------------------------------------------------
# Timeouts, resets, robustness
# ---------------------------------------------------------------------------


def test_primed_times_out_then_recovers() -> None:
    r = make_runner()
    r.feed(seg(REST, 1.0))
    r.feed(three_taps())
    r.feed(seg(REST, 9.0))  # do nothing during round 2
    kinds = r.event_kinds()
    assert kinds[:3] == [Event.ARMED, Event.PRIMED, Event.ABORTED]
    # It re-arms on its own (pose still ok) and a fresh handshake works.
    r.feed(three_taps())
    assert r.event_kinds()[-1] == Event.PRIMED


def test_torque_on_resets_everything() -> None:
    r = make_runner()
    r.feed(seg(REST, 1.0))
    r.feed(tap() + tap())
    r.feed(seg(TOUCH, 0.1) + seg(REST, 0.3), torque_off=False)  # torque during tap 3
    assert r.machine.state == "idle"
    n_before = len(r.events)
    r.feed(three_taps(), torque_off=False)  # taps while torque on: inert
    assert len(r.events) == n_before


def test_bounces_within_min_spacing_count_once() -> None:
    r = make_runner()
    r.feed(seg(REST, 1.0))
    bounce = seg(TOUCH, 0.04) + seg(REST, 0.04) + seg(TOUCH, 0.10) + seg(REST, 0.4)
    r.feed(bounce + bounce)  # 4 raw onsets, but only 2 legitimate taps
    assert Event.PRIMED not in r.event_kinds()
    r.feed(bounce)  # 3rd tap
    assert r.event_kinds()[-1] == Event.PRIMED


def test_handshake_is_repeatable() -> None:
    r = make_runner()
    for _ in range(2):
        r.feed(seg(REST, 3.5))  # cooldown + settle
        r.feed(three_taps())
        r.feed(seg(REST, 0.5))
        r.feed(three_taps())
    assert r.event_kinds().count(Event.ACTION_TAPS) == 2


# ---------------------------------------------------------------------------
# Knock counter (Counter B): pressure peaks during sustained contact.
# The recorded gestures show the user may keep the antennas together and
# "knock" without fully releasing, which edge counting cannot see.
# ---------------------------------------------------------------------------


def diff_seg(diff: float, seconds: float) -> list[tuple[float, float]]:
    return seg((diff / 2.0, -diff / 2.0), seconds)


def three_knocks_no_release() -> list[tuple[float, float]]:
    """One sustained contact with 3 pressure peaks (prominence ~0.5 rad)."""
    return (
        diff_seg(2.5, 0.2)  # contact onset = knock 1
        + diff_seg(2.0, 0.2)
        + diff_seg(2.9, 0.2)  # knock 2
        + diff_seg(2.2, 0.2)
        + diff_seg(3.0, 0.2)  # knock 3
        + seg(REST, 0.3)
    )


def make_knock_runner() -> Runner:
    return Runner(HandshakeStateMachine(HandshakeConfig(counter="knock")))


def test_knock_mode_primes_without_release() -> None:
    r = make_knock_runner()
    r.feed(seg(REST, 1.0))
    r.feed(three_knocks_no_release())
    assert r.event_kinds() == [Event.ARMED, Event.PRIMED]


def test_edge_mode_ignores_knocks() -> None:
    r = make_runner()  # default: edge counter
    r.feed(seg(REST, 1.0))
    r.feed(three_knocks_no_release())
    assert Event.PRIMED not in r.event_kinds()


def test_knock_mode_still_counts_separated_taps() -> None:
    r = make_knock_runner()
    r.feed(seg(REST, 1.0))
    r.feed(three_taps())
    assert r.event_kinds() == [Event.ARMED, Event.PRIMED]
    r.feed(seg(REST, 0.5))
    r.feed(three_taps())
    assert r.event_kinds()[-1] == Event.ACTION_TAPS


def test_knock_mode_steady_hold_still_triggers_hold_action() -> None:
    r = make_knock_runner()
    r.feed(seg(REST, 1.0))
    r.feed(three_taps())
    r.feed(seg(REST, 0.5))
    r.feed(seg(TOUCH, 1.6))  # steady pressure: no knocks, just a hold
    assert r.event_kinds() == [Event.ARMED, Event.PRIMED, Event.ACTION_HOLD]


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
