"""Unit tests for the secret-handshake detector (daemon module).

Pure and offline: synthetic 50 Hz antenna samples shaped like the real
gesture (validated against the recordings in RemiFabre/secret-handshake,
see examples/secret_handshake_lab/). No hardware, no daemon.
"""

import math

import numpy as np

from reachy_mini.daemon.backend.secret_handshake import (
    CollisionConfig,
    CollisionDetector,
    Event,
    HandshakeConfig,
    SecretHandshake,
)

DT = 0.02  # 50 Hz, the daemon control loop period

SLEEP_HEAD_POSE = np.array(
    [
        [0.911, 0.004, 0.413, -0.021],
        [-0.004, 1.0, -0.001, 0.001],
        [-0.413, -0.001, 0.911, -0.044],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
WAKE_HEAD_POSE = np.eye(4)


def d2r(deg: float) -> float:
    return math.radians(deg)


# Characteristic antenna configurations, degrees (left, right):
REST = (-12.0, 10.0)  # floppy rest: sum in band but l fails condition 2
CONTACT = (60.0, -65.0)  # touching at center: in the collision region
PRESS = (65.0, -30.0)  # firm press, flexed out of the band
CROSSED = (149.0, -160.0)  # slid past each other and parked: sum ~ -11


def seg(sample, seconds):
    return [(d2r(sample[0]), d2r(sample[1]))] * max(1, round(seconds / DT))


def tap(release_s=0.3):
    """One collision: into the band, press through, release."""
    return seg(CONTACT, 0.04) + seg(PRESS, 0.06) + seg(CONTACT, 0.04) + seg(REST, release_s)


def taps(n, release_s=0.3):
    out = []
    for _ in range(n):
        out += tap(release_s)
    return out


class Runner:
    def __init__(self, config=None):
        self.hs = SecretHandshake(config)
        self.t = 0.0
        self.events = []

    def feed(self, samples, pose=SLEEP_HEAD_POSE, torque_off=True):
        for ant0, ant1 in samples:
            e = self.hs.update(self.t, ant0, ant1, pose, torque_off)
            if e is not None:
                self.events.append(e)
            self.t += DT


# ---------------------------------------------------------------------------
# Collision detector: the geometric law
# ---------------------------------------------------------------------------


def run_detector(samples, config=None):
    det = CollisionDetector(config or CollisionConfig())
    onsets, t = 0, 0.0
    for a0, a1 in samples:
        if det.update(t, a0, a1):
            onsets += 1
        t += DT
    return onsets


def test_one_collision_per_tap():
    assert run_detector(seg(REST, 0.5) + taps(3)) == 3


def test_press_through_band_counts_once():
    assert run_detector(seg(REST, 0.5) + tap()) == 1


def test_rest_single_antenna_and_crossed_are_inert():
    assert run_detector(seg(REST, 2.0)) == 0
    assert run_detector(seg(CROSSED, 2.0)) == 0
    # only the left antenna swings, the right stays at rest
    swing = [(d2r(l), d2r(10.0)) for l in range(-170, 171, 2)]
    assert run_detector(swing) == 0


# ---------------------------------------------------------------------------
# Full handshake through the daemon-facing facade
# ---------------------------------------------------------------------------


def test_full_default_handshake():
    r = Runner()
    r.feed(seg(REST, 1.0) + taps(3))
    assert r.events == [Event.ARMED, Event.PRIMED]
    r.feed(seg(REST, 0.5) + taps(3))
    assert r.events == [Event.ARMED, Event.PRIMED, Event.SUCCESS]


def test_does_not_arm_when_head_not_in_sleep_pose():
    r = Runner()
    r.feed(seg(REST, 2.0), pose=WAKE_HEAD_POSE)
    assert r.events == []


def test_inert_while_torque_on():
    r = Runner()
    r.feed(seg(REST, 1.0) + taps(6), torque_off=False)
    assert r.events == []


def test_torque_on_mid_gesture_resets():
    r = Runner()
    r.feed(seg(REST, 1.0) + taps(2))
    r.feed(tap(), torque_off=False)
    r.feed(seg(REST, 0.2) + tap())
    # the sequence died with torque; one fresh tap must not prime
    assert Event.PRIMED not in r.events


def test_gap_of_one_second_resets_the_count():
    r = Runner()
    r.feed(seg(REST, 1.0) + taps(2))
    r.feed(seg(REST, 1.2))  # 1 s without a collision -> count back to 0
    r.feed(tap())
    assert Event.PRIMED not in r.events
    r.feed(taps(2))
    assert r.events[-1] == Event.PRIMED  # 3 quick ones prime


def test_primed_times_out_with_abort():
    r = Runner()
    r.feed(seg(REST, 1.0) + taps(3))
    r.feed(seg(REST, 3.5))  # 3 s round-2 window
    assert Event.ABORTED in r.events
    assert Event.SUCCESS not in r.events
    # after the abort it re-arms on its own (pose still ok, torque still off)
    assert r.events[-1] == Event.ARMED


def test_handshake_is_repeatable():
    r = Runner()
    for _ in range(2):
        r.feed(seg(REST, 3.5))  # cooldown + settle
        r.feed(taps(3) + seg(REST, 0.5) + taps(3))
    assert r.events.count(Event.SUCCESS) == 2


def test_never_raises_on_weird_inputs():
    hs = SecretHandshake(HandshakeConfig())
    assert hs.update(0.0, 0.0, 0.0, np.zeros((4, 4)), True) is None
    assert hs.update(0.0, 1e9, -1e9, SLEEP_HEAD_POSE, True) is None
