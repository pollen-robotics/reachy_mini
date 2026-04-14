"""V2: Multi-band energy — split audio into low/mid/high, each drives different DOFs.

Low frequencies (< 300 Hz, fundamental/prosody) → pitch (nodding)
Mid frequencies (300–2000 Hz, vowel formants) → yaw (side-to-side)
High frequencies (> 2000 Hz, sibilants/fricatives) → roll (small tilt jitter)

Translational axes get a blend. No VAD — energy directly drives movement.
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
# Tunables (v2)
# ---------------------------------------------------------------------------

DB_LOW = -60.0
DB_HIGH = -6.0
LOUDNESS_GAMMA = 0.5

ATTACK_COEFF = 0.8
RELEASE_COEFF = 0.15

# Band boundaries (Hz)
BAND_LOW_HIGH = 300
BAND_MID_HIGH = 2000

# Per-band movement amplitudes
# Low band → pitch (nods with prosody)
A_PITCH_DEG = 6.0
F_PITCH = 1.8

# Mid band → yaw (sways with vowel energy)
A_YAW_DEG = 9.0
F_YAW = 0.55

# High band → roll (jitter on sibilants)
A_ROLL_DEG = 3.0
F_ROLL = 2.5

# Translation: blend of all bands
A_X_MM = 4.0
A_Y_MM = 3.5
A_Z_MM = 2.0
F_X = 0.35
F_Y = 0.45
F_Z = 0.25


def _loudness_gain(db: float) -> float:
    t = (db - DB_LOW) / (DB_HIGH - DB_LOW)
    t = max(0.0, min(1.0, t))
    return t ** LOUDNESS_GAMMA


def _bandpass_energy(frame: NDArray[np.float32], lo_hz: float, hi_hz: float) -> float:
    """RMS energy of a frequency band via FFT."""
    n = len(frame)
    fft = np.fft.rfft(frame * np.hanning(n))
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    mask = (freqs >= lo_hz) & (freqs < hi_hz)
    band_power = np.sum(np.abs(fft[mask]) ** 2)
    rms = np.sqrt(band_power / n + 1e-12)
    return float(20.0 * math.log10(float(rms) + 1e-12))


class SwayRollRT:
    """V2: Multi-band wobbler — different frequency bands drive different DOFs."""

    def __init__(self, rng_seed: int = 7) -> None:
        self._seed = int(rng_seed)
        self.samples: deque[float] = deque(maxlen=10 * SR)
        self.carry: NDArray[np.float32] = np.zeros(0, dtype=np.float32)

        self.env_low = 0.0
        self.env_mid = 0.0
        self.env_high = 0.0

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
        self.env_low = 0.0
        self.env_mid = 0.0
        self.env_high = 0.0
        self.t = 0.0

    def _smooth(self, current: float, target: float) -> float:
        if target > current:
            return current + ATTACK_COEFF * (target - current)
        return current + RELEASE_COEFF * (target - current)

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

            # Per-band energy
            db_low = _bandpass_energy(frame, 50, BAND_LOW_HIGH)
            db_mid = _bandpass_energy(frame, BAND_LOW_HIGH, BAND_MID_HIGH)
            db_high = _bandpass_energy(frame, BAND_MID_HIGH, SR / 2)

            gain_low = _loudness_gain(db_low)
            gain_mid = _loudness_gain(db_mid)
            gain_high = _loudness_gain(db_high)

            self.env_low = self._smooth(self.env_low, gain_low)
            self.env_mid = self._smooth(self.env_mid, gain_mid)
            self.env_high = self._smooth(self.env_high, gain_high)

            self.t += HOP_MS / 1000.0

            # Low band → pitch
            pitch = (
                math.radians(A_PITCH_DEG)
                * self.env_low
                * math.sin(2 * math.pi * F_PITCH * self.t + self.phase_pitch)
            )
            # Mid band → yaw
            yaw = (
                math.radians(A_YAW_DEG)
                * self.env_mid
                * math.sin(2 * math.pi * F_YAW * self.t + self.phase_yaw)
            )
            # High band → roll
            roll = (
                math.radians(A_ROLL_DEG)
                * self.env_high
                * math.sin(2 * math.pi * F_ROLL * self.t + self.phase_roll)
            )
            # Translation: blend of all bands
            total_env = (self.env_low + self.env_mid + self.env_high) / 3.0
            x_mm = A_X_MM * total_env * math.sin(2 * math.pi * F_X * self.t + self.phase_x)
            y_mm = A_Y_MM * total_env * math.sin(2 * math.pi * F_Y * self.t + self.phase_y)
            z_mm = A_Z_MM * total_env * math.sin(2 * math.pi * F_Z * self.t + self.phase_z)

            out.append({
                "pitch_rad": pitch,
                "yaw_rad": yaw,
                "roll_rad": roll,
                "x_mm": x_mm,
                "y_mm": y_mm,
                "z_mm": z_mm,
            })

        return out
