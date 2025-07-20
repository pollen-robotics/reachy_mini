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

# ───────── GM names + MIDI helpers ──────────────────────────────────
AVAILABLE_PARTS = [
    "timpani_half",
    "synth_str_2",
    "glockenspiel",
    "strings_sp1",
    "flute_gold",
    "pan_flute",
    "bright_piano",
    "piano_merlin",
    "slow_strings_sp",
    "french_horns",
    "orchestra",
    "jazz",
    "tr_808",
    "power",
    "standard",
    "tremolande",
    "harp_lp2",
    "pizzicato_strings",
    "steel_guitar_ph",
    "alto_sax",
    "baritone_sax",
    "tenor_sax_new",
    "guitar_nylon_x",
    "harmonica",
    "viola_lp",
    "soprano_sax",
    "jazz_guitar",
    "xylophone",
    "marimba",
    "synth_strings_1",
    "church_organ_2",
    "contrabass",
    "cello_lp",
    "violin_lp3",
    "oboe",
    "clarinet",
    "bassoon_rea",
    "english_horn_rea",
    "piano_3",
    "honky_tonk",
    "e_piano_1",
    "e_piano_2",
    "clavinet",
    "celesta",
    "music_box",
    "vibraphone",
    "tubular_bells",
    "dulcimer",
    "organ_1",
    "organ_2",
    "organ_3",
    "reed_organ",
    "accordion",
    "bandoneon",
    "clean_guitar",
    "guitar_mutes",
    "overdrive_guitar",
    "distortion_guitar",
    "guitar_harmonics",
    "acoustic_bass",
    "fingered_bass",
    "picked_bass",
    "fretless_bass",
    "slap_bass_1",
    "slap_bass_2",
    "synth_bass_1",
    "synth_bass_2",
    "choir_aahs",
    "voice_oohs",
    "synth_vox",
    "orchestra_hit",
    "trumpet",
    "trombone",
    "tuba",
    "mute_trumpet",
    "brass",
    "synth_brass_1",
    "synth_brass_2",
    "piccolo",
    "recorder",
    "bottle_chiff",
    "shakuhachi",
    "whistle",
    "ocarina",
    "square_wave",
    "saw_wave",
    "synth_calliope",
    "chiffer_lead",
    "charang",
    "solo_vox",
    "fifth_saw_wave",
    "bass_and_lead",
    "fantasia",
    "warm_pad",
    "poly_synth",
    "space_voice",
    "bowed_glass",
    "metal_pad",
    "halo_pad",
    "sweep_pad",
    "ice_rain",
    "soundtrack",
    "crystal",
    "atmosphere",
    "brightness",
    "goblin",
    "echo_drops",
    "star_theme",
    "sitar",
    "banjo",
    "shamisen",
    "koto",
    "kalimba",
    "bagpipe",
    "fiddle",
    "shenai",
    "tinker_bell",
    "agogo",
    "steel_drum",
    "wood_block",
    "taiko_drum",
    "melodic_tom",
    "synth_drum",
    "reverse_cymbal",
    "fret_noise",
    "breath_noise",
    "seashore",
    "bird",
    "telephone",
    "helicopter",
    "applause",
    "gun_shot",
    "coupled_harpsichord",
]

NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]


def midi_to_name(m: int) -> str:
    return NOTE_NAMES[m % 12] + str(m // 12 - 1)


def deg_to_roman(d: int) -> str:
    romans = ["I", "II", "III", "IV", "V", "VI", "VII"]
    return romans[d] if 0 <= d < len(romans) else str(d)


# ───────── diagnostic helpers ───────────────────────────────────────
def _col_bar(val: float, lo: float, hi: float, width=24) -> str:
    """Return a coloured ASCII bar for val in [lo,hi]."""
    span = hi - lo
    pos = max(0, min(val - lo, span))
    filled = int(pos / span * width)
    empty = width - filled
    if pos / span < 0.5:
        colour = "32"  # green
    elif pos / span < 0.8:
        colour = "33"  # yellow
    else:
        colour = "31"  # red
    return f"\x1b[0;{colour}m{'█' * filled}{'-' * empty}\x1b[0m"


def _aff(x, a, b, y0, y1):
    return y0 if x <= a else y1 if x >= b else (x - a) * (y1 - y0) / (b - a) + y0


# ─────────── SCAMP session & bigger thread pool  ─────────────────────
POOL = 1000
sess = Session(max_threads=POOL)


PAD = sess.new_part("warm_pad")  # GM 89
ARP = sess.new_part("synth_bass_1")  # GM 39

ARP = sess.new_part(AVAILABLE_PARTS[2])  # GM 39


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
        self.current_root: int = 48
        self.current_pitches: list[int] = []
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
        self.current_root = root
        self.current_pitches = pitches

        self.chord_handle = PAD.start_chord(
            pitches, int(amp * 90)
        )  # velocity positional
        self.arp_notes = pitches

        self.step += 1
        self.next_change = now + 2  # hold 2 s

    # --- change instrument on the fly -----------------------

    def set_instrument(self, prog: int):
        """Change preset immediately: stop chord, switch program, restart chord."""
        global ARP, PAD
        if self.chord_handle:
            try:
                self.chord_handle.end()  # stop sound NOW
            except Exception:
                pass
            self.chord_handle = None

        # PAD.program = prog
        # ARP.program = prog
        ARP = sess.new_part(AVAILABLE_PARTS[prog])
        PAD = sess.new_part(AVAILABLE_PARTS[prog])
        # next tick() call will start the new chord with the new sound


# ───────── bar helper (numeric + coloured bar) ──────────
def _dash_bar(val: float, lo: float, hi: float, width: int = 26) -> str:
    """Return '   123.4 |███--------' style string."""
    span = hi - lo
    ratio = max(0.0, min((val - lo) / span, 1.0))
    filled = int(ratio * width)
    empty = width - filled
    if ratio < 0.5:
        colour = "32"  # green
    elif ratio < 0.8:
        colour = "33"  # yellow
    else:
        colour = "31"  # red
    bar = f"\x1b[{colour}m{'█' * filled}{'-' * empty}\x1b[0m"
    return f"{val:8.2f} |{bar}"


# ─────────── sensor loop (no queues needed) ──────────────────────────
def sensor_loop():
    cm = ChordMachine()
    sess.tempo = 120
    current_prog = -1
    current_seq = -1

    hdr = (
        "Left ant roll → BPM",
        "Right ant roll → GM‑program",
        "Head yaw → Arp speed",
        "Head pitch → Progression",
        "Head Z → Pad amp",
        "Head Y → Octave",
        "Head yaw+base → Transpose",
        "(Head roll unused)",
    )
    print("\n".join(f"{h:<30}" for h in hdr), end="\n\n")

    with ReachyMini() as r:
        r.disable_motors()
        while True:
            head, ants = r._get_current_joint_positions()
            base = head[0]
            pose = r.head_kinematics.fk(head)
            trans = pose[:3, 3] * 1_000
            rpy = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)

            # 1 Tempo (left antenna)
            bpm = int(_aff(ants[0], *LEFT_ANT_RANGE, BPM_MIN, BPM_MAX))
            sess.tempo = bpm

            # 2 Instrument (right antenna) → 0‑127 GM program
            prog = int(_aff(ants[1], *RIGHT_ANT_RANGE, 0, 127.999))
            if prog != current_prog:
                cm.set_instrument(prog)
                current_prog = prog

            # 3 Arp speed (yaw) 1–8 nps
            yaw_deg = rpy[2] + math.degrees(base)
            nps = _aff(yaw_deg, -30, 30, 1, 8)
            cm.set_arp_speed(nps)

            # 4 Progression (pitch) choose list index 0‑3
            seq_idx = int(_aff(rpy[1], -10, 10, 0, len(PROG) - 0.001))
            if seq_idx != current_seq:
                cm.seq = PROG[seq_idx]
                current_seq = seq_idx

            # 5 Pad amp (Z)
            amp = _aff(trans[2], -15, 5, 0, 1)

            # 6 Octave (Y)
            octv = int(_aff(trans[1], -10, 10, 0, 3))

            # 7 Transpose (kept on yaw+base like before)
            tr = int(_aff(yaw_deg, -10, 10, -7, 7))

            cm.tick(tr, current_prog % 4, octv, amp, bright=2000)  # brightness const

            # -------- state HUD ------------------------------------
            degree = cm.seq[cm.step % len(cm.seq)]
            chord_name = midi_to_name(cm.current_root)
            col_descr = ["5th", "add9", "sus4", "maj7"][current_prog % 4]
            notes_str = ", ".join(midi_to_name(n) for n in cm.current_pitches)

            lines = [
                f"Instrument     : {current_prog:3}  {AVAILABLE_PARTS[current_prog]}",
                f"Progression    : #{seq_idx}  [{', '.join(deg_to_roman(d) for d in cm.seq)}]",
                f"Current chord  : {chord_name} {col_descr}  ({notes_str})",
                "── Live controls ────────────────────────────────────",
                f"BPM            : {_dash_bar(bpm, BPM_MIN, BPM_MAX)}",
                f"Arp speed nps  : {_dash_bar(nps, 1, 8)}",
                f"Pad amp        : {_dash_bar(amp, 0, 1)}",
                f"Octave         : {_dash_bar(octv, 0, 3)}",
                f"Transpose st   : {_dash_bar(tr, -7, 7)}",
            ]
            sys.stdout.write("\r" + "\n".join(lines) + "\033[F" * (len(lines) - 1))
            sys.stdout.flush()
            time.sleep(0.05)


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

    """
    to get available presets, run in a terminal:
python - <<'PY'
from scamp import Session    
Session().print_default_soundfont_presets()   # prints FluidSynth’s table then exits
PY                                                                                              
   
    """
