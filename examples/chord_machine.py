#!/usr/bin/env python3
"""
Reachy Mini Groove Machine — SCAMP edition
==========================================
• Drums via NumPy + simpleaudio (exactly like before).
• Pads & arpeggio driven by FluidSynth through SCAMP (no pyo / PortAudio).
• Head pose → harmony; antennae → tempo & fills, etc. (same mapping).

Works with Reachy Mini SDK ≥ 1.0 and Python 3.10+.

Author: Remi’s debugging session, July 2025
"""

from __future__ import annotations
import math, signal, sys, threading, time, queue
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
from reachy_mini import ReachyMini

# ─────────── SCAMP / FluidSynth setup ──────────────────────────────────
from scamp import Session, wait_for_children_to_finish

# ---- audio constants --------------------
RATE = 44_100  # <‑‑ set to SCAMP’s default
SLICE_SEC = 0.05
NS = int(RATE * SLICE_SEC)

# ---- SCAMP session ----------------------
from scamp import Session

sess = Session()
PAD = sess.new_part("warm_pad")  # GM program 89
ARP = sess.new_part("synth_bass_1")  # GM program 39

# ─────────── constants & helpers ───────────────────────────────────────
SLICE_SEC = 0.05
NS = int(RATE * SLICE_SEC)

LEFT_ANT_RANGE = RIGHT_ANT_RANGE = (-math.pi, math.pi)
BASE_YAW_RANGE = (-2.0, 2.0)
BPM_MIN, BPM_MAX = 60, 200


def _aff(x, a, b, y0, y1):
    return y0 if x <= a else y1 if x >= b else (x - a) * (y1 - y0) / (b - a) + y0


# ─────────── drums (unchanged) ─────────────────────────────────────────
import simpleaudio as sa

KICK = (
    np.sin(2 * np.pi * 80 * np.linspace(0, 0.2, int(0.2 * RATE)))
    * np.exp(-np.linspace(0, 0.2, int(0.2 * RATE)) / 0.15)
).astype(np.float32)
SNARE = (
    np.random.randn(int(0.18 * RATE))
    * np.exp(-np.linspace(0, 0.18, int(0.18 * RATE)) / 0.05)
).astype(np.float32)


def build_drum_loop(pat: int, bpm: int) -> np.ndarray:
    q = 60 / bpm
    ln = int(RATE * q * 4)
    buf = np.zeros(ln, np.float32)

    def put(sample, beat, amp=1.0):
        start = int(beat * q * RATE)
        buf[start : start + sample.size] += amp * sample[: max(0, ln - start)]

    put(KICK, 0)
    put(KICK, 2)
    put(SNARE, 1)
    put(SNARE, 3)
    if pat >= 1:
        [put(SNARE, b, 0.4) for b in np.arange(0.5, 4, 1)]
    if pat >= 2:
        [put(KICK, b, 0.6) for b in (1, 3)]
    if pat >= 3:
        [put(SNARE, b, 0.3) for b in np.arange(0.25, 4, 0.5)]
    if pat >= 4:
        [put(KICK, b, 0.4) for b in np.arange(0, 4, 0.5)]
    buf /= max(1, np.max(np.abs(buf))) * 0.8
    return buf


# ─────────── harmony machinery on SCAMP ────────────────────────────────
MAJ = [0, 2, 4, 5, 7, 9, 11]
COLOUR = [[0, 7], [0, 2, 7], [0, 5, 7], [0, 2, 4, 7]]
PROG = [[0, 4, 5, 3], [0, 3, 4, 3], [5, 4, 3, 4], [0, 5, 4, 3]]


class ChordMachine:
    def __init__(self):
        self.key = 48  # C3 midi
        self.seq = PROG[0]
        self.step = 0
        self.next = 0.0  # wall‑clock seconds
        self.current_notes = None  # handles returned by start_chord
        self.arp_speed = 4  # notes per second
        self.arp_notes: list[int] = [self.key]
        self.running = True
        threading.Thread(target=self._arp_loop, daemon=True).start()

    # --- arpeggio loop --------------------------------------------------
    def _arp_loop(self):
        while self.running:
            if self.arp_notes:
                for n in self.arp_notes:
                    ARP.play_note(n, 0.9, 0.6)  # dur (beats) auto‑scaled by tempo
                    time.sleep(1 / self.arp_speed)
            else:
                time.sleep(0.05)

    # --- external setters ----------------------------------------------
    def set_arp_speed(self, nps: float):
        self.arp_speed = max(1, nps)

    # ─── inside class ChordMachine ────────────────────────────────────
    def tick(self, tr: int, col: int, octv: int, amp: float, bright: float):
        """
        Update pad / arpeggio in response to pose‑driven parameters.
        Compatible with SCAMP 0.9.x (start_chord → ChordHandle).
        """
        now = time.time()

        # --- continuous controls (filter brightness & expression) ----
        bright_val = int(_aff(bright, 300, 8000, 20, 120))
        vol_val = int(amp * 127)
        if hasattr(PAD, "cc"):
            PAD.cc(74, bright_val)  # brightness
            PAD.cc(11, vol_val)  # expression
        elif hasattr(PAD, "send_midi_cc"):
            PAD.send_midi_cc(74, bright_val / 127)
            PAD.send_midi_cc(11, vol_val / 127)

        # --- hold current chord until it's time to change ------------
        if now < self.next:
            return

        # --- finish previous chord -----------------------------------
        if self.current_notes:
            try:
                self.current_notes.end()  # ChordHandle.end()
            except Exception:
                pass
            self.current_notes = None

        # --- construct new chord -------------------------------------
        deg = self.seq[self.step % len(self.seq)]
        root = self.key + MAJ[deg] + tr
        offs = COLOUR[col % len(COLOUR)]
        pitches = [root + o + 12 * octv for o in offs]

        # velocity is 2nd positional arg in SCAMP 0.9
        self.current_notes = PAD.start_chord(pitches, int(amp * 90))
        self.arp_notes = pitches

        self.step += 1
        self.next = now + 2  # advance in 2 s


# ─────────── queues & threads ──────────────────────────────────────────
DrumMsg = Tuple[int, int, float]
ChordMsg = Tuple[int, int, int, float, float, float]
qd: "queue.Queue[DrumMsg]" = queue.Queue(maxsize=64)
qc: "queue.Queue[ChordMsg]" = queue.Queue(maxsize=128)


class DrumThread(threading.Thread):
    """
    Builds / streams the drum loop and keeps SCAMP’s tempo in sync.
    Compatible with SCAMP 0.9 (tempo is a property).
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.start()

    def run(self):
        drum_loop = build_drum_loop(0, 120)
        idx = 0
        drum_vol = 1.0

        while True:
            # see if Reachy changed bpm / pattern / volume
            while not qd.empty():
                try:
                    bpm, pat, vol = qd.get_nowait()
                    drum_loop = build_drum_loop(pat, bpm)
                    idx = 0
                    drum_vol = vol
                    sess.tempo = bpm  # property, not method
                except queue.Empty:
                    break

            # 50 ms slice
            if idx + NS <= drum_loop.size:
                sl = drum_loop[idx : idx + NS]
            else:
                sl = np.concatenate(
                    (drum_loop[idx:], drum_loop[: NS - (drum_loop.size - idx)])
                )
            idx = (idx + NS) % drum_loop.size

            sa.play_buffer(
                (np.clip(sl * drum_vol, -1, 1) * 32767).astype(np.int16), 1, 2, RATE
            )
            time.sleep(SLICE_SEC * 0.9)


# ─────────── sensor loop (unchanged mapping) ─────────────────────────–
def sensor_loop():
    cm = ChordMachine()
    with ReachyMini() as r:
        r.disable_motors()
        while True:
            head, ants = r._get_current_joint_positions()
            base = head[0]

            # drums
            bpm = int(_aff(ants[0], *LEFT_ANT_RANGE, BPM_MIN, BPM_MAX))
            pat = int(_aff(ants[1], *RIGHT_ANT_RANGE, 0, 4.999))
            vol = _aff(base, *BASE_YAW_RANGE, 0, 1)
            try:
                qd.put_nowait((bpm, pat, vol))
            except queue.Full:
                pass

            # chords
            pose = r.head_kinematics.fk(head)
            trans = pose[:3, 3] * 1000
            rpy = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
            tr = int(_aff(rpy[2] + math.degrees(base), -10, 10, -7, 7))
            col = int(_aff(rpy[1], -10, 10, 0, 3))
            octv = int(_aff(trans[1], -10, 10, 0, 3))
            amp = _aff(trans[2], -15, 5, 0, 1)
            bright = _aff(rpy[0], -10, 10, 300, 8000)
            nps = _aff(trans[2], -15, 5, 1, 8)
            try:
                qc.put_nowait((tr, col, octv, amp, bright, nps))
            except queue.Full:
                pass

            # handle chord queue immediately (in sensor thread -> low latency)
            while not qc.empty():
                tr, col, octv, amp, bright, nps = qc.get()
                cm.set_arp_speed(nps)
                cm.tick(tr, col, octv, amp, bright)
            time.sleep(0.02)


# ─────────── main & graceful shutdown ────────────────────────────────
def _shutdown(*_):
    print("\nStopping...")
    wait_for_children_to_finish(sess)  # let SCAMP finish tails
    sys.exit(0)


def main():
    DrumThread()
    signal.signal(signal.SIGINT, _shutdown)
    sensor_loop()


if __name__ == "__main__":
    main()
