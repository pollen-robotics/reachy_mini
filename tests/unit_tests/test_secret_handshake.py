"""Unit tests for the secret-handshake orchestrator (daemon module).

Pure and offline: synthetic 50 Hz antenna samples. Two subsystems share the
one facade (see docs/superpowers/specs/2026-07-07-antenna-button-handshakes-design.md):

  torque OFF: 3 antenna collisions (sleep pose) -> Event.WAKE (wake the robot)
  torque ON : antenna-button codes -> Event.WIFI / TORQUE_OFF / EMOTION

The collision law itself (CollisionDetector) is unchanged and its tests are
kept verbatim. Button-primitive and code internals are covered exhaustively
in test_antenna_buttons.py / test_button_codes.py; here we test the facade's
routing and the collapsed single-round wake gesture.
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

BASE = (-0.1745, 0.1745)  # INIT_ANTENNAS_JOINT_POSITIONS


def d2r(deg: float) -> float:
    return math.radians(deg)


# Characteristic antenna configurations, degrees (left, right):
REST = (-12.0, 10.0)
CONTACT = (60.0, -65.0)
PRESS = (65.0, -30.0)
CROSSED = (149.0, -160.0)


def seg(sample, seconds):
    return [(d2r(sample[0]), d2r(sample[1]))] * max(1, round(seconds / DT))


def tap(release_s=0.3):
    return seg(CONTACT, 0.04) + seg(PRESS, 0.06) + seg(CONTACT, 0.04) + seg(REST, release_s)


def taps(n, release_s=0.3):
    out = []
    for _ in range(n):
        out += tap(release_s)
    return out


# Button-press samples, in radians, relative to the awake base pose.
def _hold(a0, a1, n):
    return [(a0, a1)] * n


def button_single(antenna, external, n_press=3, n_release=8):
    a = list(BASE)
    delta = 0.30
    if antenna == 0:
        a[0] = BASE[0] + (-delta if external else delta)
    else:
        a[1] = BASE[1] + (delta if external else -delta)
    return _hold(a[0], a[1], n_press) + _hold(BASE[0], BASE[1], n_release)


def button_sym(external, n_press=3, n_release=8):
    if external:
        a = (BASE[0] - 0.30, BASE[1] + 0.30)
    else:
        a = (BASE[0] + 0.30, BASE[1] - 0.30)
    return _hold(a[0], a[1], n_press) + _hold(BASE[0], BASE[1], n_release)


class Runner:
    def __init__(self, config=None):
        self.hs = SecretHandshake(config)
        self.t = 0.0
        self.events = []

    def feed(self, samples, pose=SLEEP_HEAD_POSE, torque_off=True, goal=BASE):
        for ant0, ant1 in samples:
            e = self.hs.update(
                self.t, ant0, ant1, pose, torque_off, goal[0], goal[1]
            )
            if e is not None:
                self.events.append(e)
            self.t += DT


# ---------------------------------------------------------------------------
# Collision detector: the geometric law (unchanged)
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
    swing = [(d2r(l), d2r(10.0)) for l in range(-170, 171, 2)]
    assert run_detector(swing) == 0


def test_full_turn_offsets_are_normalized():
    turn = 2 * math.pi
    for l_turns, r_turns in [(-1, 1), (1, -1), (2, 0), (0, -2)]:
        samples = [
            (a0 + l_turns * turn, a1 + r_turns * turn)
            for a0, a1 in seg(REST, 0.5) + taps(3)
        ]
        assert run_detector(samples) == 3, (l_turns, r_turns)
    crossed = [(a0 - turn, a1 + turn) for a0, a1 in seg(CROSSED, 2.0)]
    assert run_detector(crossed) == 0


def test_slow_press_double_count_is_a_known_quirk():
    slow_knock = seg(CONTACT, 0.04) + seg(PRESS, 0.4) + seg(CONTACT, 0.04)
    assert run_detector(seg(REST, 0.5) + slow_knock + seg(REST, 0.5)) == 2


def test_fast_taps_all_count():
    assert run_detector(seg(REST, 0.5) + taps(3, release_s=0.12)) == 3


# ---------------------------------------------------------------------------
# Subsystem A: collision -> single-round wake (torque OFF)
# ---------------------------------------------------------------------------


def test_three_collisions_wake_the_robot():
    r = Runner()
    r.feed(seg(REST, 1.0) + taps(3))
    assert r.events == [Event.ARMED, Event.WAKE]


def test_does_not_arm_when_head_not_in_sleep_pose():
    r = Runner()
    r.feed(seg(REST, 2.0), pose=WAKE_HEAD_POSE)
    assert r.events == []


def test_collision_path_inert_while_torque_on():
    r = Runner()
    r.feed(seg(REST, 1.0) + taps(6), torque_off=False)
    assert Event.WAKE not in r.events
    assert Event.ARMED not in r.events


def test_torque_on_mid_gesture_resets_collisions():
    r = Runner()
    r.feed(seg(REST, 1.0) + taps(2))
    r.feed(tap(), torque_off=False)
    r.feed(seg(REST, 0.2) + tap())
    assert Event.WAKE not in r.events


def test_gap_of_one_second_resets_the_count():
    r = Runner()
    r.feed(seg(REST, 1.0) + taps(2))
    r.feed(seg(REST, 1.2))
    r.feed(tap())
    assert Event.WAKE not in r.events
    r.feed(taps(2))
    assert r.events[-1] == Event.WAKE


def test_immediate_retry_after_wake():
    r = Runner()
    r.feed(seg(REST, 1.0) + taps(3))
    assert r.events[-1] == Event.WAKE
    r.feed(taps(3))  # retry immediately, no pause
    assert r.events[-1] == Event.WAKE


def test_wake_is_repeatable():
    r = Runner()
    for _ in range(2):
        r.feed(seg(REST, 3.5))
        r.feed(taps(3))
    assert r.events.count(Event.WAKE) == 2


# ---------------------------------------------------------------------------
# Subsystem B: antenna-button codes (torque ON)
# ---------------------------------------------------------------------------


def test_button_wifi_code_via_facade():
    r = Runner()
    r.feed(
        button_sym(external=True)
        + button_sym(external=False)
        + button_sym(external=True),
        torque_off=False,
    )
    assert r.events == [Event.WIFI]


def test_button_order_66_code_via_facade():
    r = Runner()
    r.feed(
        button_sym(external=True)
        + button_sym(external=True)
        + button_sym(external=True),
        torque_off=False,
    )
    assert r.events == [Event.ORDER_66]


def test_button_torque_off_code_via_facade():
    r = Runner()
    samples = (
        button_single(0, external=True) * 1
        + button_single(0, external=True)
        + button_single(0, external=True)
        + button_single(1, external=True)
        + button_single(1, external=True)
    )
    r.feed(samples, torque_off=False)
    assert r.events == [Event.TORQUE_OFF]


def test_button_emotion_code_via_facade():
    r = Runner()
    samples = (
        button_single(0, external=True)
        + button_single(1, external=True)
        + button_single(0, external=True)
        + button_single(1, external=True)
    )
    r.feed(samples, torque_off=False)
    assert r.events == [Event.EMOTION]


def test_buttons_inert_while_torque_off():
    r = Runner()
    r.feed(button_sym(external=True) + button_sym(external=False), torque_off=True)
    assert Event.WIFI not in r.events


def test_button_code_works_at_a_nonbase_goal():
    # The robot is holding a non-base antenna pose (goal != base). Presses are
    # deviations from THAT goal, so the emotion code still works there.
    goal = (0.5, -0.3)

    def hold(a0, a1, n):
        return [(a0, a1)] * n

    def one(ant, npr=3, nrl=8):
        a = list(goal)
        a[ant] = goal[ant] + (-0.30 if ant == 0 else 0.30)  # external push
        return hold(a[0], a[1], npr) + hold(goal[0], goal[1], nrl)

    r = Runner()
    r.feed(one(0) + one(1) + one(0) + one(1), torque_off=False, goal=goal)
    assert r.events == [Event.EMOTION]


def test_holding_a_nonbase_goal_fires_nothing():
    goal = (0.5, -0.3)
    r = Runner()
    r.feed([(goal[0], goal[1])] * 100, torque_off=False, goal=goal)
    assert r.events == []


def test_never_raises_on_weird_inputs():
    hs = SecretHandshake(HandshakeConfig())
    assert hs.update(0.0, 0.0, 0.0, np.zeros((4, 4)), True) is None
    assert hs.update(0.0, 1e9, -1e9, SLEEP_HEAD_POSE, True) is None
    assert hs.update(0.0, 1e9, -1e9, SLEEP_HEAD_POSE, False) is None
