#!/usr/bin/env python3
"""SCAMP demo: fixed rule note->chord mapping like D3 G3 Bb3 D4, applied to all notes."""

from __future__ import annotations
import time
import numpy as np
from scamp import Session
from scipy.spatial.transform import Rotation as R

# Diatonic triad qualities for G natural minor
# roots (pitch classes): G(7), C(0), D(2) = minor; Bb(10), Eb(3), F(5) = major
MAJOR_ROOT_PCS = {10, 3, 5}
MINOR_ROOT_PCS = {7, 0, 2}
pc = lambda m: m % 12


def place_between(pitch: int, low: int, high: int) -> int:
    """Shift pitch by octaves so low < pitch < high."""
    n = pitch
    while n >= high:
        n -= 12
    while n <= low:
        n += 12
    return n


def chord_from_melody(m: int) -> list[int]:
    """
    Build a 4-note voicing like the example D3 G3 Bb3 D4 for a melody note m.
    Pattern:
      bass = m - 12               (low fifth)
      root_base = m - 7           (triad root a perfect 5th below melody)
      triad quality from G natural minor
      third_base = root_base + (3 if minor, 4 if major)
      root = placed between bass and m
      third = placed between root and m
      top = m
    Returns ascending [bass, root, third, m].
    """
    bass = m - 12
    root_base = m - 7
    is_major = pc(root_base) in MAJOR_ROOT_PCS
    third_base = root_base + (4 if is_major else 3)

    root = place_between(root_base, bass, m)
    third = place_between(third_base, root, m)

    # optional safety to avoid sub-bass mud; comment out if you want strict m-12
    if bass < 41:  # E2
        bass += 12
        # make sure root stays above bass
        if root <= bass:
            root += 12
        if third <= root:
            third += 12

    return [bass, root, third, m]


def to_velocity(d):
    # Accept 0..1 or 0..127; return 1..127
    if d <= 1.0:
        return max(1, min(127, int(round(d * 127))))
    return max(1, min(127, int(d)))


def main():
    sess = Session()
    # choir = sess.new_part("choir_aahs")
    choir = sess.new_part("orchestra_hit")
    # choir = sess.new_part("cello")

    bpm = 140
    beat = 60.0 / bpm
    e = beat / 2.0  # eighth note seconds

    # Your melody (pitch, duration_in_eighths, dyn)
    theme = [
        (62, 3, 75),  # D4
        (67, 3, 80),  # G4
        (70, 2, 85),  # Bb4
        (60, 12, 80),  # C4
        (58, 4, 75),  # Bb3
        (57, 8, 70),  # A3
        (60, 3, 78),  # C4
        (62, 3, 82),  # D4
        (65, 2, 88),  # F4
        (58, 12, 85),  # Bb3
        (57, 4, 80),  # A3
        (55, 8, 75),  # G3
        (62, 3, 75),  # D4
        (67, 3, 80),  # G4
        (70, 2, 85),  # Bb4
        (60, 12, 80),  # C4
        (58, 4, 75),  # Bb3
        (57, 8, 70),  # A3
        (57, 3, 78),  # A3
        (55, 3, 76),  # G3
        (53, 2, 74),  # F3
        (55, 24, 80),  # G3
    ]

    for m, eighths, dyn in theme:
        chord = chord_from_melody(m)  # [bass, root, third, melody]
        v = to_velocity(dyn)
        v_low = max(1, int(v * 0.8))  # slightly softer inner voices
        v_mid = max(1, int(v * 0.9))
        h1 = choir.start_note(chord[0], v_low)
        h2 = choir.start_note(chord[1], v_mid)
        h3 = choir.start_note(chord[2], v_mid)
        h4 = choir.start_note(chord[3], v)  # melody on top
        time.sleep(eighths * e)
        h1.end()
        h2.end()
        h3.end()
        h4.end()

    print("Done.")


if __name__ == "__main__":
    main()
