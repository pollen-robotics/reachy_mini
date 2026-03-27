"""Audio-reactive sway/roll generator for head wobbling.

Analyses PCM audio in real time and produces per-hop movement parameters
(pitch, yaw, roll, x, y, z) driven by voice activity and loudness.

Ported from *reachy_mini_conversation_app*.
"""

from __future__ import annotations

import math
from collections import deque
from itertools import islice
from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
SR = 16_000
FRAME_MS = 20
HOP_MS = 50

SWAY_MASTER = 1.5
SENS_DB_OFFSET = +4.0
VAD_DB_ON = -35.0
VAD_DB_OFF = -45.0
VAD_ATTACK_MS = 40
VAD_RELEASE_MS = 250
ENV_FOLLOW_GAIN = 0.65

SWAY_F_PITCH = 2.2
SWAY_A_PITCH_DEG = 4.5
SWAY_F_YAW = 0.6
SWAY_A_YAW_DEG = 7.5
SWAY_F_ROLL = 1.3
SWAY_A_ROLL_DEG = 2.25
SWAY_F_X = 0.35
SWAY_A_X_MM = 4.5
SWAY_F_Y = 0.45
SWAY_A_Y_MM = 3.75
SWAY_F_Z = 0.25
SWAY_A_Z_MM = 2.25

SWAY_DB_LOW = -46.0
SWAY_DB_HIGH = -18.0
LOUDNESS_GAMMA = 0.9
SWAY_ATTACK_MS = 50
SWAY_RELEASE_MS = 250

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------
FRAME = int(SR * FRAME_MS / 1000)
HOP = int(SR * HOP_MS / 1000)
ATTACK_FR = max(1, int(VAD_ATTACK_MS / HOP_MS))
RELEASE_FR = max(1, int(VAD_RELEASE_MS / HOP_MS))
SWAY_ATTACK_FR = max(1, int(SWAY_ATTACK_MS / HOP_MS))
SWAY_RELEASE_FR = max(1, int(SWAY_RELEASE_MS / HOP_MS))


def _rms_dbfs(x: NDArray[np.float32]) -> float:
    """Root-mean-square in dBFS for float32 mono array in [-1,1]."""
    x = x.astype(np.float32, copy=False)
    rms = np.sqrt(np.mean(x * x, dtype=np.float32) + 1e-12, dtype=np.float32)
    return float(20.0 * math.log10(float(rms) + 1e-12))


def _loudness_gain(db: float, offset: float = SENS_DB_OFFSET) -> float:
    """Normalize dB into [0,1] with gamma; clipped to [0,1]."""
    t = (db + offset - SWAY_DB_LOW) / (SWAY_DB_HIGH - SWAY_DB_LOW)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t**LOUDNESS_GAMMA if LOUDNESS_GAMMA != 1.0 else t


def _to_float32_mono(x: NDArray[Any]) -> NDArray[np.float32]:
    """Convert arbitrary PCM array to float32 mono in [-1,1].

    Accepts shapes: (N,), (1,N), (N,1), (C,N), (N,C).
    """
    a = np.asarray(x)
    if a.ndim == 0:
        return np.zeros(0, dtype=np.float32)

    if a.ndim == 2:
        if a.shape[0] <= 8 and a.shape[0] <= a.shape[1]:
            a = np.mean(a, axis=0)
        else:
            a = np.mean(a, axis=1)
    elif a.ndim > 2:
        a = np.mean(a.reshape(a.shape[0], -1), axis=0)

    if np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float32, copy=False)
    info = np.iinfo(a.dtype)
    scale = float(max(-info.min, info.max))
    return a.astype(np.float32) / (scale if scale != 0.0 else 1.0)


def _resample_linear(x: NDArray[np.float32], sr_in: int, sr_out: int) -> NDArray[np.float32]:
    """Lightweight linear resampler for short buffers."""
    if sr_in == sr_out or x.size == 0:
        return x
    n_out = int(round(x.size * sr_out / sr_in))
    if n_out <= 1:
        return np.zeros(0, dtype=np.float32)
    t_in = np.linspace(0.0, 1.0, num=x.size, dtype=np.float32, endpoint=True)
    t_out = np.linspace(0.0, 1.0, num=n_out, dtype=np.float32, endpoint=True)
    return np.interp(t_out, t_in, x).astype(np.float32, copy=False)


class SwayRollRT:
    """Feed audio chunks and get per-hop sway outputs.

    Usage::

        rt = SwayRollRT()
        results = rt.feed(pcm_int16_or_float, sr)
        # results is a list of dicts, one per HOP_MS

    """

    def __init__(self, rng_seed: int = 7) -> None:
        """Initialize state with random oscillator phases."""
        self._seed = int(rng_seed)
        self.samples: deque[float] = deque(maxlen=10 * SR)
        self.carry: NDArray[np.float32] = np.zeros(0, dtype=np.float32)

        self.vad_on = False
        self.vad_above = 0
        self.vad_below = 0

        self.sway_env = 0.0
        self.sway_up = 0
        self.sway_down = 0

        rng = np.random.default_rng(self._seed)
        self.phase_pitch = float(rng.random() * 2 * math.pi)
        self.phase_yaw = float(rng.random() * 2 * math.pi)
        self.phase_roll = float(rng.random() * 2 * math.pi)
        self.phase_x = float(rng.random() * 2 * math.pi)
        self.phase_y = float(rng.random() * 2 * math.pi)
        self.phase_z = float(rng.random() * 2 * math.pi)
        self.t = 0.0

    def reset(self) -> None:
        """Reset state (VAD/env/buffers/time) but keep initial phases/seed."""
        self.samples.clear()
        self.carry = np.zeros(0, dtype=np.float32)
        self.vad_on = False
        self.vad_above = 0
        self.vad_below = 0
        self.sway_env = 0.0
        self.sway_up = 0
        self.sway_down = 0
        self.t = 0.0

    def feed(self, pcm: NDArray[Any], sr: int | None) -> list[dict[str, float]]:
        """Stream in a PCM chunk and return sway dicts (one per hop).

        Args:
            pcm: Audio samples as ``(N,)``, ``(C,N)`` or ``(N,C)``; int or float.
            sr: Sample rate of *pcm* (``None`` assumes internal SR).

        """
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

            # VAD with hysteresis + attack/release
            if db >= VAD_DB_ON:
                self.vad_above += 1
                self.vad_below = 0
                if not self.vad_on and self.vad_above >= ATTACK_FR:
                    self.vad_on = True
            elif db <= VAD_DB_OFF:
                self.vad_below += 1
                self.vad_above = 0
                if self.vad_on and self.vad_below >= RELEASE_FR:
                    self.vad_on = False

            if self.vad_on:
                self.sway_up = min(SWAY_ATTACK_FR, self.sway_up + 1)
                self.sway_down = 0
            else:
                self.sway_down = min(SWAY_RELEASE_FR, self.sway_down + 1)
                self.sway_up = 0

            up = self.sway_up / SWAY_ATTACK_FR
            down = 1.0 - (self.sway_down / SWAY_RELEASE_FR)
            target = up if self.vad_on else down
            self.sway_env += ENV_FOLLOW_GAIN * (target - self.sway_env)
            if self.sway_env < 0.0:
                self.sway_env = 0.0
            elif self.sway_env > 1.0:
                self.sway_env = 1.0

            loud = _loudness_gain(db) * SWAY_MASTER
            env = self.sway_env
            self.t += HOP_MS / 1000.0

            # Oscillators
            pitch = (
                math.radians(SWAY_A_PITCH_DEG)
                * loud
                * env
                * math.sin(2 * math.pi * SWAY_F_PITCH * self.t + self.phase_pitch)
            )
            yaw = (
                math.radians(SWAY_A_YAW_DEG)
                * loud
                * env
                * math.sin(2 * math.pi * SWAY_F_YAW * self.t + self.phase_yaw)
            )
            roll = (
                math.radians(SWAY_A_ROLL_DEG)
                * loud
                * env
                * math.sin(2 * math.pi * SWAY_F_ROLL * self.t + self.phase_roll)
            )
            x_mm = SWAY_A_X_MM * loud * env * math.sin(2 * math.pi * SWAY_F_X * self.t + self.phase_x)
            y_mm = SWAY_A_Y_MM * loud * env * math.sin(2 * math.pi * SWAY_F_Y * self.t + self.phase_y)
            z_mm = SWAY_A_Z_MM * loud * env * math.sin(2 * math.pi * SWAY_F_Z * self.t + self.phase_z)

            out.append(
                {
                    "pitch_rad": pitch,
                    "yaw_rad": yaw,
                    "roll_rad": roll,
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "z_mm": z_mm,
                },
            )

        return out
