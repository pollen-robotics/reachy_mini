"""Feedback sounds for the handshake lab, stdlib only.

Renders short sine beeps to wav files once, plays them non-blocking through
whatever system player exists (afplay on macOS, aplay/paplay on Linux).
These are lab placeholders; the daemon will use the robot speaker and its
bundled sounds.
"""

from __future__ import annotations

import math
import os
import shutil
import struct
import subprocess
import tempfile
import wave

# Each sound is a list of (frequency_hz, duration_s); frequency 0 = silence.
SOUNDS: dict[str, list[tuple[float, float]]] = {
    # tiny tick: the gate armed, robot is listening
    "armed": [(1175.0, 0.06)],
    # single confident beep: first half of the handshake accepted
    "primed": [(880.0, 0.18)],
    # ascending fanfare: handshake A (3 taps + 3 taps) complete
    "action_taps": [(523.3, 0.09), (659.3, 0.09), (784.0, 0.09), (1046.5, 0.30)],
    # "ta-ta-daa": handshake B (3 taps + long hold) complete, clearly different
    "action_hold": [(784.0, 0.12), (0.0, 0.06), (784.0, 0.12), (0.0, 0.06), (1318.5, 0.45)],
    # low buzz: primed round timed out
    "aborted": [(220.0, 0.25)],
}


class Beeper:
    RATE = 22050

    def __init__(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="handshake_beeps_")
        self.player = next(
            (p for p in ("afplay", "aplay", "paplay", "ffplay") if shutil.which(p)),
            None,
        )
        self.files = {name: self._render(name, notes) for name, notes in SOUNDS.items()}

    def _render(self, name: str, notes: list[tuple[float, float]]) -> str:
        path = os.path.join(self.dir, f"{name}.wav")
        fade = int(0.005 * self.RATE)  # 5 ms fade in/out, avoids clicks
        frames = bytearray()
        for freq, dur in notes:
            n = int(self.RATE * dur)
            for i in range(n):
                if freq == 0.0:
                    v = 0.0
                else:
                    env = min(1.0, i / fade, (n - 1 - i) / fade)
                    v = 0.55 * env * math.sin(2.0 * math.pi * freq * i / self.RATE)
                frames += struct.pack("<h", int(v * 32767))
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.RATE)
            w.writeframes(bytes(frames))
        return path

    def play(self, name: str, blocking: bool = False) -> None:
        if self.player is None:
            print("\a", end="", flush=True)  # terminal bell fallback
            return
        cmd = [self.player, self.files[name]]
        if self.player == "aplay":
            cmd = ["aplay", "-q", self.files[name]]
        elif self.player == "ffplay":
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", self.files[name]]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if blocking:
            proc.wait()


if __name__ == "__main__":
    # Audition all the beeps once.
    b = Beeper()
    print(f"player: {b.player}, files in {b.dir}")
    for name in SOUNDS:
        print(f"  {name}")
        b.play(name, blocking=True)
