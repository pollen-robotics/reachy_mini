"""Render the daemon handshake sounds from the tap-lab tone definitions.

The reachy_mini_tap_lab web app (RemiFabre/reachy_mini_tap_lab) synthesizes
its feedback sounds with WebAudio oscillators (lib/sounds.js). Remi picked
those sounds for the daemon, so this script renders the exact same tones
(frequencies, envelopes, glides) to wav files in the package assets:

    src/reachy_mini/assets/handshake_primed.wav    (660 -> 880 two-tone)
    src/reachy_mini/assets/handshake_success.wav   (arpeggio + octave glide)
    src/reachy_mini/assets/handshake_aborted.wav   (low sawtooth buzz)

Format matches the existing daemon sounds: 44.1 kHz, 16-bit, stereo.
Deterministic: rerunning reproduces byte-identical files.

Run from the repo root:
    python examples/secret_handshake_lab/render_daemon_sounds.py
"""

from __future__ import annotations

import math
import os
import struct
import wave

RATE = 44100
ASSETS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "reachy_mini", "assets"
)

# Each tone mirrors sounds.js _tone(freq, start, dur, {type, gain, glideTo}):
# gain 0 -> peak in 5 ms (linear), peak -> 0.001 exponential over dur.
Tone = tuple  # (freq_hz, start_s, dur_s, wave_type, gain, glide_to_hz | None)

SOUNDS: dict[str, list[Tone]] = {
    "handshake_primed": [
        (660.0, 0.00, 0.12, "sine", 0.25, None),
        (880.0, 0.12, 0.18, "sine", 0.25, None),
    ],
    "handshake_success": [
        (523.25, 0.00, 0.22, "sine", 0.22, None),
        (659.25, 0.09, 0.22, "sine", 0.22, None),
        (783.99, 0.18, 0.22, "sine", 0.22, None),
        (1046.50, 0.27, 0.22, "sine", 0.22, None),
        (1046.50, 0.36, 0.50, "sine", 0.15, 2093.0),
    ],
    "handshake_aborted": [
        (140.0, 0.00, 0.30, "sawtooth", 0.15, 90.0),
    ],
}


def render(tones: list[Tone]) -> list[float]:
    total_s = max(start + dur for _, start, dur, _, _, _ in tones) + 0.05
    buf = [0.0] * int(total_s * RATE)
    for freq, start, dur, wave_type, gain, glide_to in tones:
        n = int(dur * RATE)
        i0 = int(start * RATE)
        phase = 0.0
        for i in range(n):
            ts = i / RATE
            f = freq if glide_to is None else freq * (glide_to / freq) ** (ts / dur)
            phase += f / RATE
            if wave_type == "sawtooth":
                v = 2.0 * (phase - math.floor(phase + 0.5))
            else:
                v = math.sin(2.0 * math.pi * phase)
            attack = min(1.0, ts / 0.005)
            decay = (0.001 / gain) ** (ts / dur)  # exponentialRampToValueAtTime
            buf[i0 + i] += gain * attack * decay * v
    return buf


def write_wav(path: str, buf: list[float]) -> None:
    frames = bytearray()
    for v in buf:
        s = int(max(-1.0, min(1.0, v)) * 32767)
        frames += struct.pack("<hh", s, s)  # stereo
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(bytes(frames))


def main() -> None:
    for name, tones in SOUNDS.items():
        path = os.path.abspath(os.path.join(ASSETS, f"{name}.wav"))
        write_wav(path, render(tones))
        print("wrote", path)


if __name__ == "__main__":
    main()
