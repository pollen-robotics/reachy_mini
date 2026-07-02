"""Live secret-handshake tester: the full state machine, on the robot, with beeps.

This is the pre-daemon validation script. It runs the exact pure
HandshakeStateMachine that will later live in the control loop, feeding it
the robot's present antenna positions and head pose at 50 Hz, and plays a
sound on every milestone so you can rehearse the gesture by ear:

    put the head in the sleep pose, wait ~0.5 s   ->  tiny tick   (armed)
    3 antenna collisions (knock them together)    ->  beep        (primed)
    then, within 8 s, either:
      3 more collisions      -> handshake A fanfare  (v1: the emotion)
      rub the antennas ~1 s  -> handshake B fanfare  (future: WiFi)
    do nothing for 8 s                            ->  low buzz    (aborted)

A collision is COUPLED MOTION (both antennas moving fast at once), so it does
not matter where the floppy antennas hang or which one you swing. The status
line shows m = coupled speed: it should sit at ~0 until the antennas actually
knock (spike > 2) or rub (0.3 .. 1.2).

It does NOT move the robot and leaves torque alone (use --disable-motors to
turn torque off from here; the head will slump, which is the point: the
handshake only exists in the floppy torque-off world).

Run (robot on, antennas floppy):
    python examples/secret_handshake_lab/live_handshake_probe.py
    python examples/secret_handshake_lab/live_handshake_probe.py --no-pose-gate

Sounds play on THIS machine (afplay/aplay), not through the robot speaker;
good enough to validate the logic, the daemon will use the robot speaker.
"""

from __future__ import annotations

import argparse
import time

from beeps import Beeper
from handshake import Event, HandshakeConfig, HandshakeStateMachine
from pose_gate import head_in_sleep_pose, sleep_pose_deviation

from reachy_mini import ReachyMini

DT = 0.02  # 50 Hz, same as the daemon control loop

EVENT_SOUND = {
    Event.ARMED: "armed",
    Event.PRIMED: "primed",
    Event.ACTION_TAPS: "action_taps",
    Event.ACTION_RUB: "action_rub",
    Event.ABORTED: "aborted",
}

EVENT_MESSAGE = {
    Event.ARMED: "ARMED    head in sleep pose, do 3 collisions",
    Event.PRIMED: "PRIMED   halfway there! 3 more taps = handshake A, rub = handshake B",
    Event.ACTION_TAPS: "SUCCESS  handshake A complete (3 taps + 3 taps)",
    Event.ACTION_RUB: "SUCCESS  handshake B complete (3 taps + rub)",
    Event.ABORTED: "ABORTED  no round 2 in time, back to the start",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--no-pose-gate",
        action="store_true",
        help="skip the sleep-pose check (arm anywhere; handy on a desk)",
    )
    p.add_argument(
        "--disable-motors",
        action="store_true",
        help="turn torque off at startup (the head will gently slump)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = HandshakeConfig()
    machine = HandshakeStateMachine(cfg)
    beeper = Beeper()

    print(f"pose_gate={'off' if args.no_pose_gate else 'on'}")
    if beeper.player is None:
        print("no audio player found (afplay/aplay/paplay/ffplay), using terminal bell")
    print(
        "The gesture: sleep pose -> 3 collisions -> beep -> 3 collisions (A)\n"
        "                                               or rub ~1 s      (B)\n"
        "Ctrl-C to stop.\n"
    )

    counts: dict[Event, int] = {e: 0 for e in Event}
    primed_at: float | None = None

    with ReachyMini(media_backend="no_media") as mini:
        if args.disable_motors:
            print("disabling motors (torque off), hold the head if it is up...")
            mini.disable_motors()
            time.sleep(1.0)

        t_next = time.monotonic()
        try:
            while True:
                t = time.monotonic()
                ant0, ant1 = mini.get_present_antenna_joint_positions()
                pose = mini.get_current_head_pose()
                gate_ok = True if args.no_pose_gate else head_in_sleep_pose(pose)

                event = machine.update(
                    t, ant0, ant1, torque_off=True, head_in_sleep_pose=gate_ok
                )

                if event is not None:
                    counts[event] += 1
                    primed_at = t if event is Event.PRIMED else None
                    print(f"\r\033[K[{t:10.2f}] {EVENT_MESSAGE[event]}")
                    beeper.play(EVENT_SOUND[event])

                m = machine.detector.coupled_speed
                status = (
                    f"{machine.state.upper():6s} m={m:4.1f} taps={machine.tap_count}"
                    f" ant=({ant0:+.2f},{ant1:+.2f})"
                )
                if machine.state == "primed" and primed_at is not None:
                    left = cfg.primed_timeout_s - (t - primed_at)
                    status += f" round2_timeout={max(0.0, left):.1f}s"
                if machine.state == "idle" and not gate_ok:
                    d = sleep_pose_deviation(pose)
                    status += (
                        f"  waiting for sleep pose"
                        f" (dz={d['dz_mm']:+.0f}mm dpitch={d['dpitch_deg']:+.0f}deg)"
                    )
                print(f"\r\033[K{status}", end="", flush=True)

                t_next += DT
                delay = t_next - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    t_next = time.monotonic()  # fell behind, resync
        except KeyboardInterrupt:
            pass

    print("\n\nsession summary:")
    for e in Event:
        print(f"  {e.value:12s} x{counts[e]}")


if __name__ == "__main__":
    main()
