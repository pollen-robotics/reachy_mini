"""No-robot handshake rehearsal: drive the state machine from the keyboard.

Runs the exact same 50 Hz loop and beeps as live_handshake_probe.py, but the
"antennas" are simulated: pressing Enter on an empty line injects one
collision (a coupled-motion spike, like a real knock), and `r` + Enter
injects a ~1.4 s rub (slow coupled motion). So you can hear and feel the
whole two-round flow at your desk before going to the robot:

    Enter Enter Enter        -> beep (primed)
    Enter Enter Enter        -> handshake A fanfare
or after the primed beep:
    r + Enter                -> handshake B fanfare
    wait 8 s                 -> abort buzz

Run:
    python examples/secret_handshake_lab/sim_keyboard.py
"""

from __future__ import annotations

import select
import sys
import time

from beeps import Beeper
from handshake import Event, HandshakeConfig, HandshakeStateMachine
from live_handshake_probe import EVENT_MESSAGE, EVENT_SOUND

DT = 0.02

KNOCK_STEP = 0.08  # rad per tick for 2 ticks -> ~4 rad/s coupled spike
RUB_STEP = 0.012  # rad per tick -> ~0.6 rad/s sustained coupling
RUB_TICKS = 70  # ~1.4 s


def main() -> None:
    machine = HandshakeStateMachine(HandshakeConfig())
    beeper = Beeper()
    print(__doc__)

    # Simulated floppy antennas: they stay where they were left.
    a0, a1 = -0.175, 0.175
    knock_ticks_left = 0
    rub_ticks_left = 0
    rub_direction = 1.0
    counts: dict[Event, int] = {e: 0 for e in Event}

    t_next = time.monotonic()
    try:
        while True:
            t = time.monotonic()

            # Non-blocking keyboard poll (line-based: press Enter).
            if select.select([sys.stdin], [], [], 0)[0]:
                raw = sys.stdin.readline()
                if raw == "":  # EOF (piped input ended)
                    break
                line = raw.strip().lower()
                if line in ("q", "quit"):
                    break
                if line.startswith("r") or line.startswith("h"):
                    rub_ticks_left = RUB_TICKS
                    rub_direction = -rub_direction  # stay in range over time
                else:
                    knock_ticks_left = 2

            if knock_ticks_left > 0:
                knock_ticks_left -= 1
                a0 += KNOCK_STEP
                a1 -= KNOCK_STEP
            elif rub_ticks_left > 0:
                rub_ticks_left -= 1
                a0 += RUB_STEP * rub_direction
                a1 -= RUB_STEP * rub_direction

            event = machine.update(t, a0, a1, torque_off=True, head_in_sleep_pose=True)
            if event is not None:
                counts[event] += 1
                print(f"\r\033[K[{t:10.2f}] {EVENT_MESSAGE[event]}")
                beeper.play(EVENT_SOUND[event])

            state = f"{machine.state.upper():6s} taps={machine.tap_count}"
            print(
                f"\r\033[K{state}  (Enter=collision, r+Enter=rub, q+Enter=quit)",
                end="",
                flush=True,
            )

            t_next += DT
            delay = t_next - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                t_next = time.monotonic()
    except KeyboardInterrupt:
        pass

    print("\n\nsession summary:")
    for e in Event:
        print(f"  {e.value:12s} x{counts[e]}")


if __name__ == "__main__":
    main()
