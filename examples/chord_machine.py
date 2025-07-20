#!/usr/bin/env python3
"""
Reachy Mini Groover — SCAMP‑only minimal
----------------------------------------
Pads + arpeggio driven by FluidSynth via SCAMP 0.9.x.
No simpleaudio drums—the audio path is only FluidSynth.

• Left antenna   → tempo (60‑200 BPM)
• Right antenna  → chord colour / progression step
• Head yaw/roll  → chord transpose & brightness
• Head Z         → pad amp & arp speed
"""

from __future__ import annotations
import math, signal, sys, threading, time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R
from reachy_mini import ReachyMini
from scamp import Session

# ─────────── constants & affine helper ───────────────────────────────
RATE = 44_100  # SCAMP default
LEFT_ANT_RANGE = RIGHT_ANT_RANGE = (-math.pi, math.pi)
BASE_YAW_RANGE = (-2.0, 2.0)
BPM_MIN, BPM_MAX = 60, 200


def _aff(x, a, b, y0, y1):
    return y0 if x <= a else y1 if x >= b else (x - a) * (y1 - y0) / (b - a) + y0


# ─────────── SCAMP session & bigger thread pool  ─────────────────────
POOL = 1000
sess = Session()

# # print all attributes of the session
# print("SCAMP session attributes:", ", ".join(dir(sess)))
# # print all methods of the session
# print("SCAMP session methods:", ", ".join(m for m in dir(sess) if callable))

PAD = sess.new_part("warm_pad")  # GM 89
ARP = sess.new_part("synth_bass_1")  # GM 39

# ─────────── musical data ────────────────────────────────────────────
MAJ = [0, 2, 4, 5, 7, 9, 11]
COLOUR = [[0, 7], [0, 2, 7], [0, 5, 7], [0, 2, 4, 7]]
PROG = [[0, 4, 5, 3], [0, 3, 4, 3], [5, 4, 3, 4], [0, 5, 4, 3]]


# ─────────── chord / arp engine ──────────────────────────────────────
class ChordMachine:
    def __init__(self):
        self.key = 48  # C3
        self.seq = PROG[0]
        self.step = 0
        self.next_change = 0.0
        self.chord_handle: Optional[object] = None
        self.arp_notes: list[int] = []
        self.arp_speed = 4  # notes per second
        threading.Thread(target=self._arp_loop, daemon=True).start()

    # ---- arpeggio thread -------------------------------------------
    def _arp_loop(self):
        while True:
            if self.arp_notes:
                for p in self.arp_notes:
                    ARP.play_note(p, 0.2, 0.6)  # 0.2 beat notes
                    time.sleep(1 / self.arp_speed)
            else:
                time.sleep(0.05)

    def set_arp_speed(self, nps: float):
        self.arp_speed = max(1, nps)

    def _send_cc(self, ctrl: int, val: int):
        if hasattr(PAD, "cc"):
            PAD.cc(ctrl, val)
        elif hasattr(PAD, "send_midi_cc"):
            PAD.send_midi_cc(ctrl, val / 127)

    def tick(self, tr: int, col: int, octv: int, amp: float, bright: float):
        now = time.time()

        # continuous filter brightness & expression
        self._send_cc(74, int(_aff(bright, 300, 8000, 20, 120)))
        self._send_cc(11, int(amp * 127))

        if now < self.next_change:
            return

        # end previous chord
        if self.chord_handle:
            try:
                self.chord_handle.end()
            except Exception:
                pass
            self.chord_handle = None

        # compute new chord
        deg = self.seq[self.step % len(self.seq)]
        root = self.key + MAJ[deg] + tr
        offs = COLOUR[col % len(COLOUR)]
        pitches = [root + o + 12 * octv for o in offs]

        self.chord_handle = PAD.start_chord(
            pitches, int(amp * 90)
        )  # velocity positional
        self.arp_notes = pitches

        self.step += 1
        self.next_change = now + 2  # hold 2 s


# ─────────── sensor loop (no queues needed) ──────────────────────────
def sensor_loop():
    cm = ChordMachine()
    bpm_prev = 120
    sess.tempo = bpm_prev

    with ReachyMini() as r:
        r.disable_motors()
        while True:
            head, ants = r._get_current_joint_positions()
            base = head[0]

            # --- tempo from left antenna ---------------------------
            bpm = int(_aff(ants[0], *LEFT_ANT_RANGE, BPM_MIN, BPM_MAX))
            if bpm != bpm_prev:
                sess.tempo = bpm
                bpm_prev = bpm

            # --- harmony params from pose -------------------------
            pose = r.head_kinematics.fk(head)
            trans = pose[:3, 3] * 1000
            rpy = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)

            tr = int(_aff(rpy[2] + math.degrees(base), -10, 10, -7, 7))
            col = int(_aff(rpy[1], -10, 10, 0, 3))
            octv = int(_aff(trans[1], -10, 10, 0, 3))
            amp = _aff(trans[2], -15, 5, 0, 1)
            bright = _aff(rpy[0], -10, 10, 300, 8000)
            nps = _aff(trans[2], -15, 5, 1, 8)

            cm.set_arp_speed(nps)
            cm.tick(tr, col, octv, amp, bright)
            time.sleep(0.02)


# ─────────── graceful shutdown & main ───────────────────────────────
def _shutdown(*_):
    print("\nStopping…")
    try:
        if sess.is_recording():
            sess.stop_transcribing()
    finally:
        sys.exit(0)


def main():
    signal.signal(signal.SIGINT, _shutdown)
    sensor_loop()


if __name__ == "__main__":
    main()
