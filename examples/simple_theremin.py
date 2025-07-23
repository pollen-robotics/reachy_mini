#!/usr/bin/env python3
"""
Reachy Mini Theremin — HUD + clean note‑handling (v3)
• Head roll  → pitch (‑60° … +60° maps C3 … C6)
• Head Z     → volume (‑15 mm … +5 mm maps 0 … 100 %)
Instrument   : FluidSynth preset #3 “glockenspiel” (same as your ARP default)
"""

from __future__ import annotations
import math, signal, sys, time
from reachy_mini import ReachyMini
from scamp import Session
from scipy.spatial.transform import Rotation as R

# ─── user‑tweakable ranges ──────────────────────────────────────────
ROLL_DEG_RANGE = (-60, 60)  # head‑roll span
NOTE_MIDI_RANGE = (48, 84)  # C3–C6
Z_MM_RANGE = (-15, 5)  # farther → quieter
AMP_RANGE = (0.0, 1.0)  # mute…full

# ─── init SCAMP / FluidSynth ────────────────────────────────────────
sess = Session()
THEREMIN = sess.new_part("glockenspiel")  # AVAILABLE_PARTS[2]
note_handle: object | None = None  # current sounding note
current_pitch: int | None = None  # MIDI number of that note

# ─── helpers ────────────────────────────────────────────────────────
NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
midi_to_name = lambda m: NOTE_NAMES[m % 12] + str(m // 12 - 1)


def _aff(x, a, b, y0, y1):
    return y0 if x <= a else y1 if x >= b else (x - a) * (y1 - y0) / (b - a) + y0


def _send_cc(instr, ctrl: int, val: int):
    if hasattr(instr, "cc"):  # SCAMP ≥1.0
        instr.cc(ctrl, val)
    elif hasattr(instr, "send_midi_cc"):  # SCAMP 0.9.x
        instr.send_midi_cc(ctrl, val / 127)


def _dash_bar(val: float, lo: float, hi: float, width: int = 28) -> str:
    span = hi - lo
    ratio = max(0.0, min((val - lo) / span, 1.0))
    filled = int(ratio * width)
    colour = "32" if ratio < 0.5 else "33" if ratio < 0.8 else "31"
    bar = f"\x1b[{colour}m{'█' * filled}{'-' * (width - filled)}\x1b[0m"
    return f"{val:8.2f} |{bar}"


def _shutdown(*_):
    if note_handle:
        note_handle.end()
    print("\nTheremin stopped.")
    sys.exit(0)


# ─── main loop ──────────────────────────────────────────────────────
def main():
    global note_handle, current_pitch
    signal.signal(signal.SIGINT, _shutdown)

    print("Roll → Pitch\nZ → Volume (Expression CC‑11)\n")

    with ReachyMini() as r:
        r.disable_motors()
        while True:
            head_j, _ = r._get_current_joint_positions()
            pose = r.head_kinematics.fk(head_j)
            trans_mm = pose[:3, 3] * 1_000
            roll_deg = math.degrees(R.from_matrix(pose[:3, :3]).as_euler("xyz")[0])

            target_pitch = int(round(_aff(roll_deg, *ROLL_DEG_RANGE, *NOTE_MIDI_RANGE)))
            amp = max(0.0, min(_aff(trans_mm[2], *Z_MM_RANGE, *AMP_RANGE), 1.0))

            # — play logic --------------------------------------------------
            if amp == 0.0:
                if note_handle:
                    note_handle.end()
                    note_handle = None
                    current_pitch = None
            else:
                if note_handle is None:  # start first note
                    note_handle = THEREMIN.start_note(target_pitch, int(amp * 100))
                    current_pitch = target_pitch
                elif target_pitch != current_pitch:  # pitch change
                    note_handle.end()
                    note_handle = THEREMIN.start_note(target_pitch, int(amp * 100))
                    current_pitch = target_pitch
                else:  # same note, update vol
                    _send_cc(THEREMIN, 11, int(amp * 127))

            # — HUD --------------------------------------------------------
            lines = [
                f"Note : {midi_to_name(target_pitch):>3} ({target_pitch})   Playing: {bool(note_handle)}",
                f"Head roll  (°) : {_dash_bar(roll_deg, *ROLL_DEG_RANGE)}",
                f"Head Z     (mm): {_dash_bar(trans_mm[2], *Z_MM_RANGE)}",
                f"Amplitude (0‑1) : {_dash_bar(amp, 0, 1)}",
            ]
            sys.stdout.write("\r" + "\n".join(lines) + "\033[F" * (len(lines) - 1))
            sys.stdout.flush()

            time.sleep(0.02)  # 50 Hz


if __name__ == "__main__":
    main()
