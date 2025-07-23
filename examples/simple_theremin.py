#!/usr/bin/env python3
"""
Reachy Mini Theremin (v4)
─────────────────────────
• Head roll  → pitch   (‑60° … +60° → C3 … C6)
• Head Z     → volume  (‑30 mm … +5 mm → 0 … 100 %)
• Right ant  → GM‑program 0‑127 (changes instrument on‑the‑fly)

Depend‑ons: reachy‑mini, scamp, numpy, scipy
"""

from __future__ import annotations
import math, signal, sys, time
from reachy_mini import ReachyMini
from scamp import Session
from scipy.spatial.transform import Rotation as R
import contextlib, io


# ────────────────── mapping constants ───────────────────────────────
ROLL_DEG_RANGE = (-60, 60)  # head roll span
NOTE_MIDI_RANGE = (48, 84)  # C3–C6
Z_MM_RANGE = (-30, 5)  # farther → quieter
AMP_RANGE = (0.0, 1.0)  # mute…full
RIGHT_ANT_RANGE = (-math.pi, math.pi)  # raw radians → 0‑127 program

# ────────────────── FluidSynth preset names (GM bank) ───────────────
AVAILABLE_PARTS = [
    # 0‑15
    "piano_1",
    "piano_2",
    "piano_3",
    "honky_tonk",
    "e_piano_1",
    "e_piano_2",
    "harpsichord",
    "clavinet",
    "celesta",
    "glockenspiel",
    "music_box",
    "vibraphone",
    "marimba",
    "xylophone",
    "tubular_bells",
    "dulcimer",
    # 16‑31
    "organ_1",
    "organ_2",
    "organ_3",
    "reed_organ",
    "accordion",
    "harmonica",
    "bandoneon",
    "nylon_guitar",
    "steel_guitar",
    "jazz_guitar",
    "clean_guitar",
    "muted_guitar",
    "overdrive_guitar",
    "distortion_guitar",
    "guitar_harmonics",
    "acoustic_bass",
    # 32‑47
    "fingered_bass",
    "picked_bass",
    "fretless_bass",
    "slap_bass_1",
    "slap_bass_2",
    "synth_bass_1",
    "synth_bass_2",
    "violin",
    "viola",
    "cello",
    "contrabass",
    "tremolo_strings",
    "pizzicato_strings",
    "harp",
    "timpani",
    "string_ensemble_1",
    # 48‑63
    "string_ensemble_2",
    "synth_strings_1",
    "synth_strings_2",
    "choir_aahs",
    "voice_oohs",
    "synth_voice",
    "orchestra_hit",
    "trumpet",
    "trombone",
    "tuba",
    "muted_trumpet",
    "french_horn",
    "brass_section",
    "synth_brass_1",
    "synth_brass_2",
    "soprano_sax",
    # 64‑79
    "alto_sax",
    "tenor_sax",
    "baritone_sax",
    "oboe",
    "english_horn",
    "bassoon",
    "clarinet",
    "piccolo",
    "flute",
    "recorder",
    "pan_flute",
    "blown_bottle",
    "shakuhachi",
    "whistle",
    "ocarina",
    "square_wave",
    # 80‑95
    "saw_wave",
    "synth_calliope",
    "chiffer_lead",
    "charang",
    "solo_voice",
    "fifth_saw",
    "bass_lead",
    "fantasia",
    "warm_pad",
    "polysynth",
    "space_voice",
    "bowed_glass",
    "metal_pad",
    "halo_pad",
    "sweep_pad",
    "ice_rain",
    # 96‑111
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
    "shanai",
    "tinker_bell",
    # 112‑127
    "agogo",
    "steel_drum",
    "wood_block",
    "taiko_drum",
    "melodic_tom",
    "synth_drum",
    "reverse_cymbal",
    "guitar_fret_noise",
    "breath_noise",
    "seashore",
    "bird_tweet",
    "telephone_ring",
    "helicopter",
    "applause",
    "gunshot",
    "coupled_harpsichord",
]

# ────────────────── helpers ─────────────────────────────────────────
NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
midi_to_name = lambda m: NOTE_NAMES[m % 12] + str(m // 12 - 1)


def _aff(x, a, b, y0, y1):
    return y0 if x <= a else y1 if x >= b else (x - a) * (y1 - y0) / (b - a) + y0


def _send_cc(instr, ctrl: int, val: int):
    if hasattr(instr, "cc"):
        instr.cc(ctrl, val)
    elif hasattr(instr, "send_midi_cc"):
        instr.send_midi_cc(ctrl, val / 127)


def _dash_bar(val: float, lo: float, hi: float, width=28):
    ratio = max(0.0, min((val - lo) / (hi - lo), 1.0))
    filled = int(ratio * width)
    colour = "32" if ratio < 0.5 else "33" if ratio < 0.8 else "31"
    bar = f"\x1b[{colour}m{'█' * filled}{'-' * (width - filled)}\x1b[0m"
    return f"{val:8.2f} |{bar}"


# ────────────────── session & globals ───────────────────────────────
sess = Session(max_threads=1024)
THEREMIN = sess.new_part("glockenspiel")
note_handle: object | None = None
current_pitch: int | None = None
current_prog: int | None = None
parts_cache: dict[int, "ScampInstrument"] = {}


def get_part_for_prog(prog: int):
    if prog not in parts_cache:
        buf = io.StringIO()  # swallow SCAMP prints
        with contextlib.redirect_stdout(buf):
            parts_cache[prog] = sess.new_part(AVAILABLE_PARTS[prog])

    return parts_cache[prog]


def _shutdown(*_):
    if note_handle:
        note_handle.end()
    print("\nTheremin stopped.")
    sys.exit(0)


# ────────────────── main loop ───────────────────────────────────────
def main():
    global THEREMIN, note_handle, current_pitch, current_prog
    signal.signal(signal.SIGINT, _shutdown)

    print("Roll → Pitch\nZ → Volume\nRight ant → Instrument\n")

    with ReachyMini() as r:
        r.disable_motors()
        while True:
            head_j, ants = r._get_current_joint_positions()
            pose = r.head_kinematics.fk(head_j)
            trans_mm = pose[:3, 3] * 1_000
            roll_deg = math.degrees(R.from_matrix(pose[:3, :3]).as_euler("xyz")[0])

            # mappings
            target_pitch = int(round(_aff(roll_deg, *ROLL_DEG_RANGE, *NOTE_MIDI_RANGE)))
            amp = max(0.0, min(_aff(trans_mm[2], *Z_MM_RANGE, *AMP_RANGE), 1.0))

            prog = int(_aff(ants[1], *RIGHT_ANT_RANGE, 0, 127.999))
            if prog != current_prog:
                if note_handle:
                    note_handle.end()
                    note_handle = None
                    current_pitch = None
                THEREMIN = get_part_for_prog(prog)
                current_prog = prog

            # play logic
            if amp == 0.0:
                if note_handle:
                    note_handle.end()
                    note_handle = None
                    current_pitch = None
            else:
                if note_handle is None:
                    note_handle = THEREMIN.start_note(target_pitch, int(amp * 100))
                    current_pitch = target_pitch
                elif target_pitch != current_pitch:
                    note_handle.end()
                    note_handle = THEREMIN.start_note(target_pitch, int(amp * 100))
                    current_pitch = target_pitch
                else:
                    _send_cc(THEREMIN, 11, int(amp * 127))

            # HUD
            lines = [
                f"Instrument     : {current_prog if current_prog is not None else '--':>3} "
                f"{AVAILABLE_PARTS[current_prog] if current_prog is not None else ''}",
                f"Note           : {midi_to_name(target_pitch):>3} ({target_pitch})   Playing: {bool(note_handle)}",
                f"Head roll  (°) : {_dash_bar(roll_deg, *ROLL_DEG_RANGE)}",
                f"Head Z     (mm): {_dash_bar(trans_mm[2], *Z_MM_RANGE)}",
                f"Amplitude (0‑1): {_dash_bar(amp, 0, 1)}",
                f"Right ant (rad): {_dash_bar(ants[1], *RIGHT_ANT_RANGE)}",
            ]
            sys.stdout.write("\r" + "\n".join(lines) + "\033[F" * (len(lines) - 1))
            sys.stdout.flush()

            time.sleep(0.02)  # 50 Hz loop


if __name__ == "__main__":
    main()
