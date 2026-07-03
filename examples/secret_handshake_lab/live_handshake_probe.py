"""Live secret-handshake tester: a faithful simulation of the daemon loop.

The core of this script is shaped EXACTLY like the future daemon
integration: a 50 Hz control loop that reads what the daemon already reads
(present antenna positions, head pose, torque state) and makes ONE call:

    event = handshake.update(t, ant0, ant1, head_pose, torque_off)

Everything else here (beeps, status printing) is lab-only feedback driven by
the returned event, exactly like the daemon will drive the robot speaker and
the action. All tunables live in HandshakeConfig / CollisionConfig; see the
banner at the top of handshake.py for the values at a glance.

The gesture:

    put the head in the sleep pose, wait ~0.5 s   ->  tiny tick   (armed)
    3 antenna collisions in quick succession      ->  beep        (primed)
    (a 1 s pause between collisions resets the count)
    then, within 3 s, either:
      3 more collisions            -> handshake A fanfare  (v1: the emotion)
      hold them gently together    -> handshake B fanfare  (future: WiFi)
    or nothing                     -> low buzz, back to the start

It disables torque at startup (hold the head if it is up: it will slump,
which is the point: the handshake only exists in the floppy torque-off
world). Pass --keep-torque to leave the motors as they are. It never moves
the robot.

NOTE on latency: the state machine reacts on the exact tick (sub-us, see
bench.py). What feels slow here is the TEMPORARY sound path: each beep
spawns an afplay/aplay process on this computer (~150-400 ms to open the
audio device). The daemon will play through the robot speaker instead.

Run (robot on, antennas floppy):
    python examples/secret_handshake_lab/live_handshake_probe.py
    python examples/secret_handshake_lab/live_handshake_probe.py --no-pose-gate
"""

from __future__ import annotations

import argparse
import time

from beeps import Beeper
from handshake import Event, HandshakeConfig, SecretHandshake
from pose_gate import SLEEP_HEAD_POSE, head_in_sleep_pose, sleep_pose_deviation

from reachy_mini import ReachyMini

DT = 0.02  # 50 Hz, same as the daemon control loop

EVENT_SOUND = {
    Event.ARMED: "armed",
    Event.PRIMED: "primed",
    Event.ACTION_TAPS: "action_taps",
    Event.ACTION_HOLD: "action_hold",
    Event.ABORTED: "aborted",
}

EVENT_MESSAGE = {
    Event.ARMED: "ARMED    head in sleep pose, do 3 quick collisions",
    Event.PRIMED: "PRIMED   halfway there! 3 more taps = handshake A, gentle hold = handshake B",
    Event.ACTION_TAPS: "SUCCESS  handshake A complete (3 taps + 3 taps)",
    Event.ACTION_HOLD: "SUCCESS  handshake B complete (3 taps + hold)",
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
        "--keep-torque",
        action="store_true",
        help="do not touch the motors at startup (default: disable them)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = HandshakeConfig()
    handshake = SecretHandshake(cfg)
    beeper = Beeper()
    det = handshake.machine.detector  # read-only, for the status line

    print(
        f"pose_gate={'off' if args.no_pose_gate else 'on'}\n"
        f"collision: sum in [{det.sum_lo_deg:.0f}, {det.sum_hi_deg:.0f}] deg "
        f"AND l in [{cfg.collision.l_min_deg:.0f}, {cfg.collision.l_max_deg:.0f}] deg\n"
        f"timing: {cfg.taps_required} collisions, max {cfg.max_gap_s:.0f} s apart; "
        f"round 2 within {cfg.primed_timeout_s:.0f} s; hold = {cfg.hold_min_s:.0f} s\n"
    )
    if beeper.player is None:
        print("no audio player found (afplay/aplay/paplay/ffplay), using terminal bell")
    print("Ctrl-C to stop.\n")

    counts: dict[Event, int] = {e: 0 for e in Event}
    primed_at: float | None = None

    with ReachyMini(media_backend="no_media") as mini:
        if not args.keep_torque:
            print("disabling motors (torque off), hold the head if it is up...")
            mini.disable_motors()
            time.sleep(1.0)

        t_next = time.monotonic()
        try:
            while True:
                # ---- the daemon-shaped part: read sensors, ONE call --------
                t = time.monotonic()
                ant0, ant1 = mini.get_present_antenna_joint_positions()
                head_pose = (
                    SLEEP_HEAD_POSE if args.no_pose_gate else mini.get_current_head_pose()
                )
                event = handshake.update(t, ant0, ant1, head_pose, torque_off=True)
                # ------------------------------------------------------------

                if event is not None:
                    counts[event] += 1
                    primed_at = t if event is Event.PRIMED else None
                    print(f"\r\033[K[{t:10.2f}] {EVENT_MESSAGE[event]}")
                    beeper.play(EVENT_SOUND[event])

                machine = handshake.machine
                band = "IN-BAND" if det.in_collision else "       "
                status = (
                    f"{machine.state.upper():6s} sum={det.sum_deg:+6.1f}deg "
                    f"l={det.l_deg:+6.1f}deg {band} taps={machine.tap_count}"
                )
                if machine.state == "primed" and primed_at is not None:
                    left = cfg.primed_timeout_s - (t - primed_at)
                    status += f" round2_timeout={max(0.0, left):.1f}s"
                if machine.state == "idle" and not args.no_pose_gate:
                    if not head_in_sleep_pose(head_pose):
                        d = sleep_pose_deviation(head_pose)
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
