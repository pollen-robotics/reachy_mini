"""Unit tests for the coupled-motion collision law and the handshake machine.

Everything here is synthetic and offline: we generate 50 Hz antenna samples
that mimic the real signals (validated against the HF recordings, see
analyze_recordings.py) and feed them tick by tick, exactly like the daemon
control loop will.

The physical model behind the synthetic traces:
- The antennas are floppy friction-fit parts: they stay where they are left.
  Absolute angles mean nothing; only motion does.
- A collision (knock) moves BOTH antennas fast for a couple of ticks
  (coupled speed ~4 rad/s in the recordings).
- A rub/slide moves both antennas slowly and continuously (~0.3-1.2 rad/s).
- Moving a single antenna, however fast, is not a collision.

Run:
    python -m pytest examples/secret_handshake_lab/test_handshake.py -v
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collision import CollisionConfig, CollisionDetector  # noqa: E402
from handshake import Event, HandshakeConfig, HandshakeStateMachine  # noqa: E402
from pose_gate import SLEEP_HEAD_POSE, head_in_sleep_pose  # noqa: E402

DT = 0.02  # 50 Hz, same as the daemon control loop


class Trace:
    """Builds a synthetic antenna trajectory, stateful like real antennas."""

    def __init__(self, a0: float = -0.175, a1: float = 0.175) -> None:
        self.a0 = a0
        self.a1 = a1
        self.samples: list[tuple[float, float]] = []

    def _emit(self) -> None:
        self.samples.append((self.a0, self.a1))

    def still(self, seconds: float) -> "Trace":
        for _ in range(max(1, round(seconds / DT))):
            self._emit()
        return self

    def knock(self, settle_s: float = 0.35) -> "Trace":
        """One collision: both antennas jump fast (~4 rad/s) for 2 ticks."""
        for _ in range(2):
            self.a0 += 0.08
            self.a1 -= 0.08
            self._emit()
        return self.still(settle_s)

    def knocks(self, n: int, settle_s: float = 0.35) -> "Trace":
        for _ in range(n):
            self.knock(settle_s)
        return self

    def rub(self, seconds: float, step: float = 0.012) -> "Trace":
        """Touching and sliding: both move slowly (~0.6 rad/s), no spikes."""
        for i in range(max(1, round(seconds / DT))):
            self.a0 += step
            self.a1 -= step
            self._emit()
        return self

    def move_one(self, seconds: float, step: float = 0.08) -> "Trace":
        """Only ant0 moves (fast). Not a collision, must be inert."""
        for _ in range(max(1, round(seconds / DT))):
            self.a0 += step
            self._emit()
        return self


class Runner:
    """Feeds samples to a machine with a monotonic clock, collects events."""

    def __init__(self, machine: HandshakeStateMachine) -> None:
        self.machine = machine
        self.t = 0.0
        self.events: list[tuple[float, Event]] = []

    def feed(
        self,
        trace: Trace,
        torque_off: bool = True,
        pose_ok: bool = True,
    ) -> list[Event]:
        new: list[Event] = []
        for ant0, ant1 in trace.samples:
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


def continued(trace_runner: Runner) -> Trace:
    """A fresh Trace starting where the runner's last sample left off."""
    # Tests build several traces in a row; antennas must not teleport between
    # them or the jump itself would register as motion.
    last = trace_runner.machine.last_sample
    if last is None:
        return Trace()
    return Trace(last[0], last[1])


# ---------------------------------------------------------------------------
# Collision detector: the coupled-motion law
# ---------------------------------------------------------------------------


def run_detector(trace: Trace) -> tuple[int, int]:
    det = CollisionDetector(CollisionConfig())
    knocks = 0
    coupled_ticks = 0
    t = 0.0
    for a0, a1 in trace.samples:
        if det.update(t, a0, a1):
            knocks += 1
        if det.coupled:
            coupled_ticks += 1
        t += DT
    return knocks, coupled_ticks


def test_detector_counts_one_knock_per_impact() -> None:
    knocks, _ = run_detector(Trace().still(0.5).knocks(3).still(0.5))
    assert knocks == 3


def test_detector_refractory_merges_bounces() -> None:
    # Two impacts 100 ms apart are one collision (bounce), not two.
    knocks, _ = run_detector(Trace().still(0.5).knock(settle_s=0.06).knock())
    assert knocks == 1


def test_detector_rub_is_coupled_but_not_a_knock() -> None:
    knocks, coupled_ticks = run_detector(Trace().still(0.5).rub(1.0).still(0.3))
    assert knocks == 0
    assert coupled_ticks >= 40  # ~1 s of coupling at 50 Hz


def test_detector_ignores_single_antenna_motion() -> None:
    knocks, coupled_ticks = run_detector(Trace().still(0.5).move_one(1.0).still(0.3))
    assert knocks == 0
    assert coupled_ticks == 0


def test_detector_ignores_statics_anywhere() -> None:
    # Parked crossed (the end state of the collision-definition recordings)
    # or any other static position is not contact.
    knocks, coupled_ticks = run_detector(Trace(a0=2.6, a1=-2.8).still(2.0))
    assert knocks == 0
    assert coupled_ticks == 0


# ---------------------------------------------------------------------------
# Arming gate
# ---------------------------------------------------------------------------


def test_arms_after_settling_in_sleep_pose() -> None:
    r = make_runner()
    r.feed(Trace().still(1.0))
    assert r.event_kinds() == [Event.ARMED]
    assert r.machine.state == "armed"


def test_does_not_arm_without_sleep_pose() -> None:
    r = make_runner()
    r.feed(Trace().still(2.0), pose_ok=False)
    assert r.event_kinds() == []
    assert r.machine.state == "idle"


def test_does_not_arm_with_torque_on() -> None:
    r = make_runner()
    r.feed(Trace().still(2.0), torque_off=False)
    assert r.event_kinds() == []
    assert r.machine.state == "idle"


def test_pose_flicker_restarts_settle_timer() -> None:
    r = make_runner()
    r.feed(Trace().still(0.3))
    r.feed(Trace().still(0.1), pose_ok=False)
    r.feed(Trace().still(0.3))
    assert r.event_kinds() == []  # 0.3 s < settle time, must not be armed yet
    r.feed(Trace().still(0.4))
    assert r.event_kinds() == [Event.ARMED]


# ---------------------------------------------------------------------------
# Handshake A: 3 collisions -> beep -> 3 collisions -> action
# ---------------------------------------------------------------------------


def test_full_knock_knock_handshake() -> None:
    r = make_runner()
    r.feed(Trace().still(1.0).knocks(3))
    assert r.event_kinds() == [Event.ARMED, Event.PRIMED]
    r.feed(continued(r).still(0.5).knocks(3))
    assert r.event_kinds() == [Event.ARMED, Event.PRIMED, Event.ACTION_TAPS]
    assert r.machine.state == "idle"  # back to start after the action


def test_two_knocks_do_not_prime() -> None:
    r = make_runner()
    r.feed(Trace().still(1.0).knocks(2))
    assert r.machine.tap_count == 2  # live UI reads this
    r.feed(continued(r).still(4.0))  # let the rhythm window expire
    assert Event.PRIMED not in r.event_kinds()
    assert r.machine.tap_count == 0


def test_slow_knocks_never_prime() -> None:
    r = make_runner()
    r.feed(Trace().still(1.0).knocks(4, settle_s=1.7))  # 3 s window holds only 2
    assert Event.PRIMED not in r.event_kinds()


def test_single_antenna_motion_is_inert() -> None:
    # The old diff law false-positived on this; the coupled law must not.
    r = make_runner()
    r.feed(Trace().still(1.0))
    r.feed(continued(r).move_one(0.2).still(0.3).move_one(0.2).still(0.3).move_one(0.2))
    assert r.event_kinds() == [Event.ARMED]
    assert r.machine.tap_count == 0


def test_works_at_any_antenna_base_position() -> None:
    # The live robot rested at diff=+0.97 (dataset robot: -0.35). The law
    # must not care where the floppy antennas happen to hang.
    r = Runner(HandshakeStateMachine(HandshakeConfig()))
    r.feed(Trace(a0=0.606, a1=-0.367).still(1.0).knocks(3))
    assert r.event_kinds() == [Event.ARMED, Event.PRIMED]


# ---------------------------------------------------------------------------
# Handshake B: 3 collisions -> beep -> rub/slide -> other action
# ---------------------------------------------------------------------------


def test_rub_handshake() -> None:
    r = make_runner()
    r.feed(Trace().still(1.0).knocks(3))
    r.feed(continued(r).still(0.5).rub(1.6))
    assert r.event_kinds() == [Event.ARMED, Event.PRIMED, Event.ACTION_RUB]
    assert r.machine.state == "idle"


def test_rub_in_round_one_does_not_prime() -> None:
    r = make_runner()
    r.feed(Trace().still(1.0).rub(2.0).still(0.5))
    assert r.event_kinds() == [Event.ARMED]


def test_knocks_reset_the_rub_timer() -> None:
    # Knock-rub-knock-rub in round 2 must not add up to a rub action.
    r = make_runner()
    r.feed(Trace().still(1.0).knocks(3))
    r.feed(continued(r).still(0.4).knock().rub(0.7).knock().rub(0.7))
    assert Event.ACTION_RUB not in r.event_kinds()


# ---------------------------------------------------------------------------
# Timeouts, resets, robustness
# ---------------------------------------------------------------------------


def test_primed_times_out_then_recovers() -> None:
    r = make_runner()
    r.feed(Trace().still(1.0).knocks(3))
    r.feed(continued(r).still(9.0))  # do nothing during round 2
    kinds = r.event_kinds()
    assert kinds[:3] == [Event.ARMED, Event.PRIMED, Event.ABORTED]
    # It re-arms on its own (pose still ok) and a fresh handshake works.
    r.feed(continued(r).knocks(3))
    assert r.event_kinds()[-1] == Event.PRIMED


def test_torque_on_resets_everything() -> None:
    r = make_runner()
    r.feed(Trace().still(1.0).knocks(2))
    r.feed(continued(r).knock(), torque_off=False)  # torque during collision 3
    assert r.machine.state == "idle"
    n_before = len(r.events)
    r.feed(continued(r).knocks(3), torque_off=False)  # knocks while torque on
    assert len(r.events) == n_before


def test_handshake_is_repeatable() -> None:
    r = make_runner()
    for _ in range(2):
        r.feed(continued(r).still(3.5))  # cooldown + settle
        r.feed(continued(r).knocks(3))
        r.feed(continued(r).still(0.5).knocks(3))
    assert r.event_kinds().count(Event.ACTION_TAPS) == 2


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
