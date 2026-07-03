"""No-robot handshake rehearsal: drive the state machine from the keyboard.

Runs the exact same 50 Hz loop and beeps as live_handshake_probe.py, but the
"antennas" are simulated: pressing Enter on an empty line injects one
collision (a quick pass through the geometric collision band, like a real
knock), and `h` + Enter injects a ~1.4 s gentle hold inside the band. So you
can hear and feel the whole two-round flow at your desk before going to the
robot:

    Enter Enter Enter        -> beep (primed)
    Enter Enter Enter        -> handshake A fanfare
or after the primed beep:
    h + Enter                -> handshake B fanfare
    wait 8 s                 -> abort buzz

Run:
    python examples/secret_handshake_lab/sim_keyboard.py
"""

from __future__ import annotations

import math
import select
import sys
import time

from beeps import Beeper
from handshake import Event, HandshakeConfig, SecretHandshake
from live_handshake_probe import EVENT_MESSAGE, EVENT_SOUND
from pose_gate import SLEEP_HEAD_POSE

DT = 0.02


def d2r(deg: float) -> float:
    return math.radians(deg)


# Characteristic configurations (degrees), matching the real geometry:
REST = (d2r(-12.0), d2r(10.0))  # sum in band but l fails condition 2
CONTACT = (d2r(60.0), d2r(-65.0))  # in the collision band
PRESS = (d2r(65.0), d2r(-30.0))  # flexed through the band

# One tap = in through the band, press, back through, release.
TAP_PROFILE = [CONTACT] * 2 + [PRESS] * 3 + [CONTACT] * 2
HOLD_TICKS = 70  # ~1.4 s inside the band


def main() -> None:
    handshake = SecretHandshake(HandshakeConfig())
    beeper = Beeper()
    print(__doc__)

    pending: list[tuple[float, float]] = []  # samples queued by keypresses
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
                if line.startswith("h"):
                    pending = [CONTACT] * HOLD_TICKS
                else:
                    pending = list(TAP_PROFILE)

            ant0, ant1 = pending.pop(0) if pending else REST
            event = handshake.update(
                t, ant0, ant1, head_pose=SLEEP_HEAD_POSE, torque_off=True
            )
            if event is not None:
                counts[event] += 1
                print(f"\r\033[K[{t:10.2f}] {EVENT_MESSAGE[event]}")
                beeper.play(EVENT_SOUND[event])

            state = f"{handshake.machine.state.upper():6s} taps={handshake.machine.tap_count}"
            print(
                f"\r\033[K{state}  (Enter=collision, h+Enter=hold, q+Enter=quit)",
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
