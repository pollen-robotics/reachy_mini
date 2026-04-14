"""V3: Onset impulse — energy onsets trigger decaying movement impulses.

Detects sudden rises in audio energy (syllable onsets) and fires a
short movement impulse that decays exponentially. Each onset picks a
random direction, so the movement pattern is non-repetitive and
naturally synced to speech rhythm.

A background envelope still provides gentle sway during sustained sound.
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
# Tunables (v3)
# ---------------------------------------------------------------------------

DB_LOW = -60.0
DB_HIGH = -6.0
LOUDNESS_GAMMA = 0.5

# Envelope for background sway
ENV_ATTACK = 0.6
ENV_RELEASE = 0.1

# Onset detection
ONSET_THRESHOLD = 0.15  # minimum gain jump to trigger an onset
ONSET_COOLDOWN_HOPS = 3  # minimum hops between onsets (~150ms)

# Impulse decay
IMPULSE_DECAY = 0.85  # per-hop multiplier (0.85^20 ≈ 0.04, so ~1s decay)

# Background sway amplitudes (gentle, always-on when there's energy)
BG_A_PITCH_DEG = 2.0
BG_A_YAW_DEG = 3.0
BG_F_PITCH = 1.5
BG_F_YAW = 0.4

# Impulse amplitudes (triggered on onsets)
IMPULSE_A_PITCH_DEG = 6.0
IMPULSE_A_YAW_DEG = 10.0
IMPULSE_A_ROLL_DEG = 4.0
IMPULSE_A_X_MM = 5.0
IMPULSE_A_Y_MM = 4.0
IMPULSE_A_Z_MM = 2.5


def _loudness_gain(db: float) -> float:
    t = (db - DB_LOW) / (DB_HIGH - DB_LOW)
    t = max(0.0, min(1.0, t))
    return t ** LOUDNESS_GAMMA


class SwayRollRT:
    """V3: Onset impulse wobbler — syllable onsets trigger decaying movements."""

    def __init__(self, rng_seed: int = 7) -> None:
        self._seed = int(rng_seed)
        self.samples: deque[float] = deque(maxlen=10 * SR)
        self.carry: NDArray[np.float32] = np.zeros(0, dtype=np.float32)

        self.envelope = 0.0
        self.prev_gain = 0.0
        self.cooldown = 0

        # Active impulses: list of (strength, decay, direction_vector)
        # direction_vector is (pitch, yaw, roll, x, y, z) with random signs
        self.impulses: list[tuple[float, NDArray[np.float64]]] = []

        self.rng = np.random.default_rng(self._seed)
        self.phase_pitch = float(self.rng.random() * 2 * math.pi)
        self.phase_yaw = float(self.rng.random() * 2 * math.pi)
        self.t = 0.0

    def reset(self) -> None:
        self.samples.clear()
        self.carry = np.zeros(0, dtype=np.float32)
        self.envelope = 0.0
        self.prev_gain = 0.0
        self.cooldown = 0
        self.impulses.clear()
        self.t = 0.0

    def _spawn_impulse(self, strength: float) -> None:
        """Create a new impulse with random direction."""
        # Random direction: each axis gets a random sign and magnitude
        direction = self.rng.uniform(-1.0, 1.0, size=6)
        # Normalize so the total magnitude is consistent
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        self.impulses.append((strength, direction))

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

            # Smooth envelope for background sway
            if gain > self.envelope:
                self.envelope += ENV_ATTACK * (gain - self.envelope)
            else:
                self.envelope += ENV_RELEASE * (gain - self.envelope)

            # Onset detection: sudden rise in gain
            if self.cooldown > 0:
                self.cooldown -= 1

            delta = gain - self.prev_gain
            if delta > ONSET_THRESHOLD and self.cooldown == 0:
                self._spawn_impulse(min(delta * 3.0, 1.0))
                self.cooldown = ONSET_COOLDOWN_HOPS

            self.prev_gain = gain
            self.t += HOP_MS / 1000.0

            # Sum impulse contributions
            imp_pitch = 0.0
            imp_yaw = 0.0
            imp_roll = 0.0
            imp_x = 0.0
            imp_y = 0.0
            imp_z = 0.0

            surviving: list[tuple[float, NDArray[np.float64]]] = []
            for strength, direction in self.impulses:
                imp_pitch += strength * direction[0] * math.radians(IMPULSE_A_PITCH_DEG)
                imp_yaw += strength * direction[1] * math.radians(IMPULSE_A_YAW_DEG)
                imp_roll += strength * direction[2] * math.radians(IMPULSE_A_ROLL_DEG)
                imp_x += strength * direction[3] * IMPULSE_A_X_MM
                imp_y += strength * direction[4] * IMPULSE_A_Y_MM
                imp_z += strength * direction[5] * IMPULSE_A_Z_MM
                new_strength = strength * IMPULSE_DECAY
                if new_strength > 0.01:
                    surviving.append((new_strength, direction))
            self.impulses = surviving

            # Background sway (gentle, always on when there's energy)
            env = self.envelope
            bg_pitch = (
                math.radians(BG_A_PITCH_DEG)
                * env
                * math.sin(2 * math.pi * BG_F_PITCH * self.t + self.phase_pitch)
            )
            bg_yaw = (
                math.radians(BG_A_YAW_DEG)
                * env
                * math.sin(2 * math.pi * BG_F_YAW * self.t + self.phase_yaw)
            )

            out.append({
                "pitch_rad": bg_pitch + imp_pitch,
                "yaw_rad": bg_yaw + imp_yaw,
                "roll_rad": imp_roll,
                "x_mm": imp_x,
                "y_mm": imp_y,
                "z_mm": imp_z,
            })

        return out
