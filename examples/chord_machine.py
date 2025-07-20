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
    (mini) remi@KamaEndure:~/reachy_mini/examples$ python - <<'PY'
from scamp import Session    
Session().print_default_soundfont_presets()   # prints FluidSynth’s table then exits
PY                                                                                              
WARNING:root:python-rtmidi was not found; streaming midi input / output will not be available.
PRESETS FOR default (general_midi)
   Preset[000:047] Timpani Half 2 bag(s) from #0
   Preset[000:051] Synth Str 2 3 bag(s) from #2
   Preset[000:009] Glockenspiel 2 bag(s) from #5
   Preset[000:048] Strings SP1 2 bag(s) from #7
   Preset[000:073] Flute Gold 2 bag(s) from #9
   Preset[000:075] Pan Flute 2 bag(s) from #11
   Preset[000:001] Bright Piano 2 bag(s) from #13
   Preset[000:000] Piano Merlin 6 bag(s) from #15
   Preset[000:049] Slow Strings SP 2 bag(s) from #21
   Preset[000:060] French Horns 2 bag(s) from #23
   Preset[128:048] Orchestra 2 bag(s) from #25
   Preset[128:032] Jazz 2 bag(s) from #27
   Preset[128:025] TR 808 2 bag(s) from #29
   Preset[128:016] Power 2 bag(s) from #31
   Preset[128:000] Standard 2 bag(s) from #33
   Preset[000:044] Tremolande 2 bag(s) from #35
   Preset[000:046] Harp LP2 2 bag(s) from #37
   Preset[000:045] Pizzicato Strings 2 bag(s) from #39
   Preset[000:025] Steel Guitar PH 2 bag(s) from #41
   Preset[000:065] Alto SAX 2 bag(s) from #43
   Preset[000:067] Baritone Sax 2 bag(s) from #45
   Preset[000:066] Tenor Sax New 2 bag(s) from #47
   Preset[000:024] Guitar Nylon X 2 bag(s) from #49
   Preset[000:022] Harmonica 2 bag(s) from #51
   Preset[000:041] Viola LP 2 bag(s) from #53
   Preset[000:064] Soprano Sax 3 bag(s) from #55
   Preset[000:026] Jazz Guitar 2 bag(s) from #58
   Preset[000:013] Xylophone 3 bag(s) from #60
   Preset[000:012] Marimba 2 bag(s) from #63
   Preset[000:050] Synth Strings 1 2 bag(s) from #65
   Preset[000:019] Church Organ 2 4 bag(s) from #67
   Preset[000:043] Contrabass 2 bag(s) from #71
   Preset[000:042] Cello LP 3 bag(s) from #73
   Preset[000:040] Violin LP3 2 bag(s) from #76
   Preset[000:068] Oboe 2 bag(s) from #78
   Preset[000:071] Clarinet 2 bag(s) from #80
   Preset[000:070] Bassoon (Rea) 2 bag(s) from #82
   Preset[000:069] English Horn (Rea) 2 bag(s) from #84
   Preset[000:002] Piano 3 3 bag(s) from #86
   Preset[000:003] Honky Tonk 3 bag(s) from #89
   Preset[000:004] E.Piano 1 3 bag(s) from #92
   Preset[000:005] E.Piano 2 2 bag(s) from #95
   Preset[000:007] Clavinet 2 bag(s) from #97
   Preset[000:008] Celesta 2 bag(s) from #99
   Preset[000:010] Music Box 2 bag(s) from #101
   Preset[000:011] Vibraphone 2 bag(s) from #103
   Preset[000:014] Tubular Bells 2 bag(s) from #105
   Preset[000:015] Dulcimer 2 bag(s) from #107
   Preset[000:016] Organ 1 2 bag(s) from #109
   Preset[000:017] Organ 2 2 bag(s) from #111
   Preset[000:018] Organ 3 3 bag(s) from #113
   Preset[000:020] Reed Organ 3 bag(s) from #116
   Preset[000:021] Accordion 3 bag(s) from #119
   Preset[000:023] Bandoneon 2 bag(s) from #122
   Preset[000:027] Clean Guitar 2 bag(s) from #124
   Preset[000:028] Guitar Mutes 2 bag(s) from #126
   Preset[000:029] Overdrive Guitar 2 bag(s) from #128
   Preset[000:030] DistortionGuitar 2 bag(s) from #130
   Preset[000:031] Guitar Harmonics 2 bag(s) from #132
   Preset[000:032] Acoustic Bass 2 bag(s) from #134
   Preset[000:033] Fingered Bass 2 bag(s) from #136
   Preset[000:034] Picked Bass 2 bag(s) from #138
   Preset[000:035] Fretless Bass 2 bag(s) from #140
   Preset[000:036] Slap Bass 1 2 bag(s) from #142
   Preset[000:037] Slap Bass 2 2 bag(s) from #144
   Preset[000:038] Synth Bass 1 2 bag(s) from #146
   Preset[000:039] Synth Bass 2 2 bag(s) from #148
   Preset[000:052] Choir Aahs 4 bag(s) from #150
   Preset[000:053] Voice Oohs 3 bag(s) from #154
   Preset[000:054] Synth Vox 2 bag(s) from #157
   Preset[000:055] Orchestra Hit 2 bag(s) from #159
   Preset[000:056] Trumpet 2 bag(s) from #161
   Preset[000:057] Trombone 2 bag(s) from #163
   Preset[000:058] Tuba 2 bag(s) from #165
   Preset[000:059] Mute Trumpet 2 bag(s) from #167
   Preset[000:061] Brass 2 bag(s) from #169
   Preset[000:062] Synth Brass 1 2 bag(s) from #171
   Preset[000:063] Synth Brass 2 2 bag(s) from #173
   Preset[000:072] Piccolo 2 bag(s) from #175
   Preset[000:074] Recorder 2 bag(s) from #177
   Preset[000:076] Bottle Chiff 2 bag(s) from #179
   Preset[000:077] Shakuhachi 2 bag(s) from #181
   Preset[000:078] Whistle 2 bag(s) from #183
   Preset[000:079] Ocarina 2 bag(s) from #185
   Preset[000:080] Square Wave 3 bag(s) from #187
   Preset[000:081] Saw Wave 3 bag(s) from #190
   Preset[000:082] Synth Calliope 2 bag(s) from #193
   Preset[000:083] Chiffer Lead 2 bag(s) from #195
   Preset[000:084] Charang 2 bag(s) from #197
   Preset[000:085] Solo Vox 2 bag(s) from #199
   Preset[000:086] 5th Saw Wave 2 bag(s) from #201
   Preset[000:087] Bass & Lead 2 bag(s) from #203
   Preset[000:088] Fantasia 5 bag(s) from #205
   Preset[000:089] Warm Pad 2 bag(s) from #210
   Preset[000:090] Poly Synth 2 bag(s) from #212
   Preset[000:091] Space Voice 2 bag(s) from #214
   Preset[000:092] Bowed Glass 3 bag(s) from #216
   Preset[000:093] Metal Pad 2 bag(s) from #219
   Preset[000:094] Halo Pad 2 bag(s) from #221
   Preset[000:095] Sweep Pad 2 bag(s) from #223
   Preset[000:096] Ice Rain 2 bag(s) from #225
   Preset[000:097] Soundtrack 2 bag(s) from #227
   Preset[000:098] Crystal 2 bag(s) from #229
   Preset[000:099] Atmosphere 2 bag(s) from #231
   Preset[000:100] Brightness 2 bag(s) from #233
   Preset[000:101] Goblin 2 bag(s) from #235
   Preset[000:102] Echo Drops 2 bag(s) from #237
   Preset[000:103] Star Theme 2 bag(s) from #239
   Preset[000:104] Sitar 2 bag(s) from #241
   Preset[000:105] Banjo 2 bag(s) from #243
   Preset[000:106] Shamisen 2 bag(s) from #245
   Preset[000:107] Koto 2 bag(s) from #247
   Preset[000:108] Kalimba 2 bag(s) from #249
   Preset[000:109] Bagpipe 2 bag(s) from #251
   Preset[000:110] Fiddle 2 bag(s) from #253
   Preset[000:111] Shenai 2 bag(s) from #255
   Preset[000:112] Tinker Bell 2 bag(s) from #257
   Preset[000:113] Agogo 2 bag(s) from #259
   Preset[000:114] Steel Drum 2 bag(s) from #261
   Preset[000:115] Wood Block 2 bag(s) from #263
   Preset[000:116] Taiko Drum 2 bag(s) from #265
   Preset[000:117] Melodic Tom 2 bag(s) from #267
   Preset[000:118] Synth Drum 2 bag(s) from #269
   Preset[000:119] Reverse Cymbal 2 bag(s) from #271
   Preset[000:120] Fret Noise 2 bag(s) from #273
   Preset[000:121] Breath Noise 2 bag(s) from #275
   Preset[000:122] Seashore 2 bag(s) from #277
   Preset[000:123] Bird 2 bag(s) from #279
   Preset[000:124] Telephone 2 bag(s) from #281
   Preset[000:125] Helicopter 2 bag(s) from #283
   Preset[000:126] Applause 2 bag(s) from #285
   Preset[000:127] Gun Shot 2 bag(s) from #287
   Preset[000:006] Coupled Harpsichord 3 bag(s) from #289
   Preset EOP

    
    """
