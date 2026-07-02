"""No-robot handshake rehearsal: drive the state machine from the keyboard.

Runs the exact same 50 Hz loop and beeps as live_handshake_probe.py, but the
"antennas" are simulated: every time you press Enter on an empty line it
injects one collision (a 0.1 s contact), and typing `h` + Enter injects a
press-and-hold. So you can hear and feel the whole two-round flow at your
desk before going to the robot:

    Enter Enter Enter        -> beep (primed)
    Enter Enter Enter        -> handshake A fanfare
or after the primed beep:
    h + Enter                -> handshake B fanfare
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

REST = (-0.175, 0.175)
TOUCH = (1.25, -1.25)

TAP_CONTACT_S = 0.10
HOLD_CONTACT_S = 1.5


def main() -> None:
    machine = HandshakeStateMachine(HandshakeConfig())
    beeper = Beeper()
    print(__doc__)

    contact_until = 0.0
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
                duration = HOLD_CONTACT_S if line.startswith("h") else TAP_CONTACT_S
                contact_until = t + duration

            ant0, ant1 = TOUCH if t < contact_until else REST
            event = machine.update(
                t, ant0, ant1, torque_off=True, head_in_sleep_pose=True
            )
            if event is not None:
                counts[event] += 1
                print(f"\r\033[K[{t:10.2f}] {EVENT_MESSAGE[event]}")
                beeper.play(EVENT_SOUND[event])

            state = f"{machine.state.upper():6s} taps={machine.tap_count}"
            print(f"\r\033[K{state}  (Enter=tap, h+Enter=hold, q+Enter=quit)", end="", flush=True)

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
