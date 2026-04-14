"""V1: Direct envelope — no VAD, energy directly drives movement amplitude.

Removes the VAD gate entirely. RMS energy is smoothed with fast attack /
medium release and mapped to [0,1] over a wide dB range. Oscillators
still provide directional variation, but their amplitude tracks the
speech energy contour directly — loud sustained vowels produce large
sustained movement, pauses produce stillness.
"""

from __future__ import annotations

import math
from collections import deque
from itertools import islice
from typing import Any

import numpy as np
from numpy.typing import NDArray

from reachy_mini.motion.speech_tapper import (
    SR,
    FRAME_MS,
    HOP_MS,
    FRAME,
    HOP,
    _rms_dbfs,
    _to_float32_mono,
    _resample_linear,
)

# ---------------------------------------------------------------------------
# Tunables (v1)
# ---------------------------------------------------------------------------

# Wider dB range so loud audio doesn't saturate
DB_LOW = -60.0
DB_HIGH = -6.0
LOUDNESS_GAMMA = 0.5  # compress dynamic range (sqrt-like)

# Envelope smoothing — fast attack to catch syllables, slower release
ATTACK_COEFF = 0.8   # how fast envelope rises (0→1: instant→never)
RELEASE_COEFF = 0.15  # how fast envelope falls (slower = longer tail)

# Movement amplitudes (same axes as v0)
A_PITCH_DEG = 5.0
A_YAW_DEG = 8.0
A_ROLL_DEG = 2.5
A_X_MM = 5.0
A_Y_MM = 4.0
A_Z_MM = 2.5

# Oscillator frequencies — provide directional variation
F_PITCH = 2.2
F_YAW = 0.6
F_ROLL = 1.3
F_X = 0.35
F_Y = 0.45
F_Z = 0.25


def _loudness_gain(db: float) -> float:
    """Map dB to [0,1] with gamma compression over the wide range."""
    t = (db - DB_LOW) / (DB_HIGH - DB_LOW)
    t = max(0.0, min(1.0, t))
    return t ** LOUDNESS_GAMMA


class SwayRollRT:
    """V1: Direct envelope wobbler — no VAD, energy = amplitude."""

    def __init__(self, rng_seed: int = 7) -> None:
        self._seed = int(rng_seed)
        self.samples: deque[float] = deque(maxlen=10 * SR)
        self.carry: NDArray[np.float32] = np.zeros(0, dtype=np.float32)

        self.envelope = 0.0  # smoothed energy envelope [0, 1]

        rng = np.random.default_rng(self._seed)
        self.phase_pitch = float(rng.random() * 2 * math.pi)
        self.phase_yaw = float(rng.random() * 2 * math.pi)
        self.phase_roll = float(rng.random() * 2 * math.pi)
        self.phase_x = float(rng.random() * 2 * math.pi)
        self.phase_y = float(rng.random() * 2 * math.pi)
        self.phase_z = float(rng.random() * 2 * math.pi)
        self.t = 0.0

    def reset(self) -> None:
        self.samples.clear()
        self.carry = np.zeros(0, dtype=np.float32)
        self.envelope = 0.0
        self.t = 0.0

    def feed(self, pcm: NDArray[Any], sr: int | None) -> list[dict[str, float]]:
        sr_in = SR if sr is None else int(sr)
        x = _to_float32_mono(pcm)
        if x.size == 0:
            return []
        if sr_in != SR:
            x = _resample_linear(x, sr_in, SR)
            if x.size == 0:
                return []

        if self.carry.size:
            self.carry = np.concatenate([self.carry, x])
        else:
            self.carry = x

        out: list[dict[str, float]] = []

        while self.carry.size >= HOP:
            hop = self.carry[:HOP]
            self.carry = self.carry[HOP:]

            self.samples.extend(hop.tolist())
            if len(self.samples) < FRAME:
                self.t += HOP_MS / 1000.0
                continue

            frame = np.fromiter(
                islice(self.samples, len(self.samples) - FRAME, len(self.samples)),
                dtype=np.float32,
                count=FRAME,
            )
            db = _rms_dbfs(frame)
            gain = _loudness_gain(db)

            # Asymmetric smoothing: fast attack, slower release
            if gain > self.envelope:
                self.envelope += ATTACK_COEFF * (gain - self.envelope)
            else:
                self.envelope += RELEASE_COEFF * (gain - self.envelope)

            env = self.envelope
            self.t += HOP_MS / 1000.0

            # Oscillators scaled directly by envelope
            pitch = (
                math.radians(A_PITCH_DEG)
                * env
                * math.sin(2 * math.pi * F_PITCH * self.t + self.phase_pitch)
            )
            yaw = (
                math.radians(A_YAW_DEG)
                * env
                * math.sin(2 * math.pi * F_YAW * self.t + self.phase_yaw)
            )
            roll = (
                math.radians(A_ROLL_DEG)
                * env
                * math.sin(2 * math.pi * F_ROLL * self.t + self.phase_roll)
            )
            x_mm = A_X_MM * env * math.sin(2 * math.pi * F_X * self.t + self.phase_x)
            y_mm = A_Y_MM * env * math.sin(2 * math.pi * F_Y * self.t + self.phase_y)
            z_mm = A_Z_MM * env * math.sin(2 * math.pi * F_Z * self.t + self.phase_z)

            out.append({
                "pitch_rad": pitch,
                "yaw_rad": yaw,
                "roll_rad": roll,
                "x_mm": x_mm,
                "y_mm": y_mm,
                "z_mm": z_mm,
            })

        return out
