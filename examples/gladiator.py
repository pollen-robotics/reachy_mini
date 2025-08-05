#!/usr/bin/env python3
"""Minimal SCAMP demo: Gladiator-inspired original motif (not the film melody)."""

from __future__ import annotations

import time
import numpy as np  # kept to match your deps
from scamp import Session
from scipy.spatial.transform import Rotation as R  # kept to match your deps


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

    # Play the theme
    for p, eighths, dyn in theme:
        h = choir.start_note(p, dyn)
        time.sleep(eighths * e)
        h.end()

    print("Done.")


if __name__ == "__main__":
    main()
