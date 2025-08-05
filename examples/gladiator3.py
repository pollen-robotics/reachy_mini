#!/usr/bin/env python3
"""Minimal SCAMP demo: Gladiator-inspired original motif (not the film melody)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np  # kept to match your deps
from pynput import keyboard
from scamp import Session
from scipy.spatial.transform import Rotation as R  # kept to match your deps


# --- small helpers ---------------------------------------------------
pc = lambda m: m % 12
MAJOR_ROOT_PCS = {10, 3, 5}  # Bb, Eb, F (major triads in G natural minor)
# G natural minor triads: Gm, Cm, Dm (minor) and Bb, Eb, F (major).


def _vol_from_dyn(d):
    # Accept 0..1 floats or ~0..127 ints; return 0..1
    return float(d) / 127.0 if d > 1 else max(0.0, min(1.0, float(d)))


def _place_between(pitch: int, low: int, high: int) -> int:
    n = pitch
    while n >= high:
        n -= 12
    while n <= low:
        n += 12
    return n


def chord_from_melody(m: int) -> list[int]:
    """
    Note->chord mapping matching the example "D3 G3 Bb3 D4" for melody D4:
      bass  = m - 12
      root  = ~a fifth below melody (m - 7), adjusted into [bass, m)
      third = minor or major third above root, chosen diatonically
      top   = m
    Returns ascending [bass, root, third, m].
    """
    bass = m - 12
    root_guess = m - 7
    is_major = pc(root_guess) in MAJOR_ROOT_PCS
    third_guess = root_guess + (4 if is_major else 3)

    root = _place_between(root_guess, bass, m)
    third = _place_between(third_guess, root, m)

    # keep very low basses out of mud; lift one octave if below ~E2
    if bass < 41:
        bass += 12
        if root <= bass:
            root += 12
        if third <= root:
            third += 12

    return [bass, root, third, m]


# --- interactive state (keyboard, thread-safe) -----------------------
@dataclass
class State:
    chord_mode: bool = False
    next_inst: bool = False
    prev_inst: bool = False
    stop: bool = False
    lock: threading.Lock = threading.Lock()

    def toggle_chords(self):
        with self.lock:
            self.chord_mode = not self.chord_mode

    def pull_flags(self):
        with self.lock:
            flags = (self.next_inst, self.prev_inst, self.chord_mode, self.stop)
            self.next_inst = False
            self.prev_inst = False
            return flags


def start_keyboard_listener(state: State):
    def on_press(key):
        if isinstance(key, keyboard.KeyCode) and key.char:
            c = key.char.lower()
            if c == "c":
                state.toggle_chords()
        elif key == keyboard.Key.right:
            with state.lock:
                state.next_inst = True
        elif key == keyboard.Key.left:
            with state.lock:
                state.prev_inst = True
        elif key in (keyboard.Key.esc, keyboard.Key.ctrl_c):
            with state.lock:
                state.stop = True
            return False  # stop listener
        return None

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener


# --- main ------------------------------------------------------------
def main():
    sess = Session()

    # choir = sess.new_part("choir_aahs")
    # choir = sess.new_part("honky_tonk")
    # choir = sess.new_part("piano_1")
    choir = sess.new_part("music_box")
    # choir = sess.new_part("vibraphone")
    # choir = sess.new_part("viola")
    # choir = sess.new_part("cello")
    # choir = sess.new_part("string_ensemble_2")
    # choir = sess.new_part("voice_oohs")
    # choir = sess.new_part("synth_voice")
    # choir = sess.new_part("piccolo")
    # choir = sess.new_part("warm_pad") # Maybe with less tremolo?

    instruments = [
        "choir_aahs",
        "honky_tonk",
        "piano_1",
        "music_box",
        "vibraphone",
        "viola",
        "cello",
        "string_ensemble_2",
        "voice_oohs",
        "synth_voice",
        "piccolo",
        "warm_pad",  # Maybe with less tremolo?
    ]

    parts_cache = {"music_box": choir}

    def get_part(name: str):
        if name not in parts_cache:
            # silence SCAMP prints if needed, but keep it simple here
            parts_cache[name] = sess.new_part(name)
        return parts_cache[name]

    cur_idx = instruments.index("music_box") if "music_box" in instruments else 0
    part = get_part(instruments[cur_idx])

    # Tempo reference
    bpm = 140  # dotted-quarter around 78 feels right for 6/8
    beat = 60.0 / bpm  # quarter note seconds
    e = beat / 2.0  # eighth note seconds

    theme = [
        # Phrase 1
        (62, 3, 75),  # D4
        (67, 3, 80),  # G4
        (70, 2, 85),  # Bb4
        (60, 12, 80),  # C4 (held long)
        (58, 4, 75),  # Bb3
        (57, 8, 70),  # A3
        # Phrase 2
        (60, 3, 78),  # C4
        (62, 3, 82),  # D4
        (65, 2, 88),  # F4
        (58, 12, 85),  # Bb3 (held long)
        (57, 4, 80),  # A3
        (55, 8, 75),  # G3
        # Phrase 3
        (62, 3, 75),  # D4
        (67, 3, 80),  # G4
        (70, 2, 85),  # Bb4
        (60, 12, 80),  # C4 (held long)
        (58, 4, 75),  # Bb3
        (57, 8, 70),  # A3
        # Phrase 4
        (57, 3, 78),  # A3
        (55, 3, 76),  # G3
        (53, 2, 74),  # F3
        (55, 24, 80),  # G3 (held very long to resolve)
    ]

    state = State()
    listener = start_keyboard_listener(state)

    try:
        for p, eighths, dyn in theme:
            # apply any pending control changes
            next_inst, prev_inst, chord_mode, stop = state.pull_flags()
            if stop:
                break
            if next_inst:
                cur_idx = (cur_idx + 1) % len(instruments)
                part = get_part(instruments[cur_idx])
            if prev_inst:
                cur_idx = (cur_idx - 1 + len(instruments)) % len(instruments)
                part = get_part(instruments[cur_idx])

            dur = eighths * e

            if chord_mode:
                chord = chord_from_melody(p)
                h = part.start_chord(chord, _vol_from_dyn(dyn))
            else:
                h = part.start_note(p, dyn)

            # simple hold with light polling so instrument/mode changes apply next note
            t0 = time.time()
            while True:
                if state.stop:
                    break
                if (time.time() - t0) >= dur:
                    break
                time.sleep(0.01)

            h.end()
            if state.stop:
                break

        print("Done.")
    finally:
        # best-effort: stop listener thread
        try:
            listener.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
