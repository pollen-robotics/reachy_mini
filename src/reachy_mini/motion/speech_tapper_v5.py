"""V5: v4 with an F0-relative pitch tilt for prosodic head nodding.

v4 already drives gestures from rising-edge syllable triggers with AGC and a
strict silence gate. v5 adds one extra signal on top: a slow-baseline F0
tracker. The head pitch axis gets a small offset proportional to how far the
current (smoothed) F0 sits above or below the speaker's running baseline,
in semitones.

Effect: rising intonation lifts the head, falling intonation drops it,
without overwhelming the discrete trigger-driven gestures inherited from v4
(yaw, roll, x, y, z are unchanged from v4; pitch is v4_pitch + tilt). The
baseline self-adapts (slow EMA, ~16 s time constant) so it works for any
speaker pitch range without manual calibration.
"""

from __future__ import annotations

import math
from collections import deque
from itertools import islice

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Tunables (v5 mirrors v4 plus an F0 tilt block)
# ---------------------------------------------------------------------------
FRAME_MS = 40
HOP_MS = 50

VAD_DB_ON = -32.0
VAD_DB_HARD_OFF = -50.0
VAD_VOICED_MEMORY_MS = 120
VAD_LOUD_MEMORY_MS = 100

F0_MIN = 80.0
F0_MAX = 400.0
VOICING_THRESHOLD = 0.40

SPEECH_LEVEL_ALPHA = 0.040
SPEECH_DEFAULT_DB = -28.0
SPEECH_UPDATE_FLOOR_DB = -42.0
ABS_QUIET_DB = -50.0
REL_DB_QUIET = -8.0
REL_DB_LOUD = +6.0
LOUDNESS_GAMMA = 0.7

ENV_ATTACK = 0.7
ENV_RELEASE = 0.3

# Floor on the envelope during VAD-active frames (same as v4).
MIN_ENVELOPE = 0.65

NUCLEUS_HISTORY_HOPS = 3
NUCLEUS_RISE_THRESHOLD = 0.08
NUCLEUS_MIN_GAIN = 0.18
NUCLEUS_MIN_SPACING_MS = 110

DIR_LERP = 0.40
DIR_DECAY = 0.88
MUTE_DECAY = 0.6

A_PITCH_DEG = 16.0
A_YAW_DEG = 28.0
A_ROLL_DEG = 10.0
A_X_MM = 16.0
A_Y_MM = 13.0
A_Z_MM = 8.0
DIR_FLOOR = 0.70
DIR_LOUD_BOOST = 0.30

# Continuous breath sine layer (same as v4)
BREATH_F_PITCH = 1.5
BREATH_A_PITCH_DEG = 2.0
BREATH_F_YAW = 0.6
BREATH_A_YAW_DEG = 4.0
BREATH_F_ROLL = 1.0
BREATH_A_ROLL_DEG = 1.3
BREATH_F_X = 0.4
BREATH_A_X_MM = 2.0
BREATH_F_Y = 0.45
BREATH_A_Y_MM = 1.6
BREATH_F_Z = 0.3
BREATH_A_Z_MM = 1.0

# F0 prosody (v5 only)
F0_REF_HZ = 100.0
F0_SMOOTH_ALPHA = 0.5                # ~100 ms smoothing on raw F0
F0_BASELINE_ALPHA = 0.020            # ~2.5 s baseline tracker (was ~16 s)
TILT_DEG_PER_SEMITONE = 1.5          # 2.5x bigger so the effect is visible
TILT_MAX_DEG = 8.0                    # 2x range
TILT_LERP = 0.40                     # 2x faster smoothing on the tilt output
TILT_SIGN = -1.0                     # flip if positive pitch_rad turns the head the unintuitive way

# Derived
VAD_VOICED_MEMORY_FR = max(1, int(VAD_VOICED_MEMORY_MS / HOP_MS))
VAD_LOUD_MEMORY_FR = max(1, int(VAD_LOUD_MEMORY_MS / HOP_MS))
NUCLEUS_MIN_SPACING_FR = max(1, int(NUCLEUS_MIN_SPACING_MS / HOP_MS))


def _rms_dbfs(x: NDArray[np.float32]) -> float:
    rms = float(np.sqrt(np.mean(x.astype(np.float32) ** 2) + 1e-12))
    return 20.0 * math.log10(rms + 1e-12)


def _voicing_and_f0(frame: NDArray[np.float32], sample_rate: int) -> tuple[bool, float]:
    """Return (voiced, f0_hz). Autocorrelation peak in F0 lag range; f0=0 if unvoiced."""
    n = len(frame)
    if n < 32:
        return False, 0.0
    f = frame - float(np.mean(frame))
    f = f * np.hanning(n).astype(np.float32)
    fft = np.fft.rfft(f, n=n * 2)
    auto = np.fft.irfft(np.abs(fft) ** 2, n=n * 2)[:n]
    if auto[0] <= 1e-9:
        return False, 0.0
    auto = auto / auto[0]
    min_lag = max(1, int(sample_rate / F0_MAX))
    max_lag = min(n - 1, int(sample_rate / F0_MIN))
    if min_lag >= max_lag:
        return False, 0.0
    region = auto[min_lag:max_lag]
    peak_idx = int(np.argmax(region))
    peak_corr = float(region[peak_idx])
    if peak_corr < VOICING_THRESHOLD:
        return False, 0.0
    if 0 < peak_idx < len(region) - 1:
        y0, y1, y2 = float(region[peak_idx - 1]), peak_corr, float(region[peak_idx + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-9:
            offset = 0.5 * (y0 - y2) / denom
        else:
            offset = 0.0
    else:
        offset = 0.0
    lag = (min_lag + peak_idx + offset)
    if lag <= 0:
        return False, 0.0
    return True, sample_rate / lag


class SwayRollRT:
    """V5: v4 with an F0-relative pitch tilt added."""

    def __init__(self, rng_seed: int = 7, sample_rate: int = 16_000) -> None:
        self._seed = int(rng_seed)
        self.sample_rate = int(sample_rate)
        self.frame = int(self.sample_rate * FRAME_MS / 1000)
        self.hop = int(self.sample_rate * HOP_MS / 1000)
        self.samples: deque[float] = deque(maxlen=10 * self.sample_rate)
        self.carry: NDArray[np.float32] = np.zeros(0, dtype=np.float32)

        self.envelope = 0.0
        self.last_voiced_age = 999
        self.last_loud_age = 999

        self.speech_db = SPEECH_DEFAULT_DB
        self.speech_db_init = False

        self.loud_history: deque[float] = deque(
            [0.0] * NUCLEUS_HISTORY_HOPS, maxlen=NUCLEUS_HISTORY_HOPS,
        )
        self.last_nucleus_age = 999

        self.target_dir = np.zeros(6, dtype=np.float32)
        self.current_dir = np.zeros(6, dtype=np.float32)

        self.f0_smoothed = 0.0
        self.f0_baseline_st = 0.0
        self.f0_baseline_init = False
        self.tilt_current_rad = 0.0

        self.rng = np.random.default_rng(self._seed)
        self.phase_pitch = float(self.rng.random() * 2 * math.pi)
        self.phase_yaw = float(self.rng.random() * 2 * math.pi)
        self.phase_roll = float(self.rng.random() * 2 * math.pi)
        self.phase_x = float(self.rng.random() * 2 * math.pi)
        self.phase_y = float(self.rng.random() * 2 * math.pi)
        self.phase_z = float(self.rng.random() * 2 * math.pi)
        self.t = 0.0

    def reset(self) -> None:
        self.samples.clear()
        self.carry = np.zeros(0, dtype=np.float32)
        self.envelope = 0.0
        self.last_voiced_age = 999
        self.last_loud_age = 999
        self.speech_db = SPEECH_DEFAULT_DB
        self.speech_db_init = False
        self.loud_history = deque(
            [0.0] * NUCLEUS_HISTORY_HOPS, maxlen=NUCLEUS_HISTORY_HOPS,
        )
        self.last_nucleus_age = 999
        self.target_dir = np.zeros(6, dtype=np.float32)
        self.current_dir = np.zeros(6, dtype=np.float32)
        self.f0_smoothed = 0.0
        self.f0_baseline_st = 0.0
        self.f0_baseline_init = False
        self.tilt_current_rad = 0.0
        self.t = 0.0

    def _loudness_gain_agc(self, db: float) -> float:
        if db < ABS_QUIET_DB:
            return 0.0
        rel = db - self.speech_db
        t = (rel - REL_DB_QUIET) / (REL_DB_LOUD - REL_DB_QUIET)
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        return float(t ** LOUDNESS_GAMMA)

    def _new_target_direction(self, prominence: float) -> NDArray[np.float32]:
        d = self.rng.uniform(-1.0, 1.0, size=6).astype(np.float32)
        d = np.sign(d) * np.abs(d) ** 1.5
        m = float(np.max(np.abs(d)))
        if m > 1e-9:
            d = d / m
        scale = DIR_FLOOR + DIR_LOUD_BOOST * float(prominence)
        return (d * scale).astype(np.float32)

    def _zero(self) -> dict[str, float]:
        return {
            "pitch_rad": 0.0,
            "yaw_rad": 0.0,
            "roll_rad": 0.0,
            "x_mm": 0.0,
            "y_mm": 0.0,
            "z_mm": 0.0,
        }

    def feed(self, pcm: NDArray[np.float32]) -> list[dict[str, float]]:
        if pcm.size == 0:
            return []
        if self.carry.size:
            self.carry = np.concatenate([self.carry, pcm])
        else:
            self.carry = pcm

        out: list[dict[str, float]] = []

        while self.carry.size >= self.hop:
            hop = self.carry[:self.hop]
            self.carry = self.carry[self.hop:]

            self.samples.extend(hop.tolist())
            if len(self.samples) < self.frame:
                self.t += HOP_MS / 1000.0
                out.append(self._zero())
                continue

            frame = np.fromiter(
                islice(self.samples, len(self.samples) - self.frame, len(self.samples)),
                dtype=np.float32,
                count=self.frame,
            )

            db = _rms_dbfs(frame)
            voiced, f0_hz = _voicing_and_f0(frame, self.sample_rate)

            self.last_voiced_age += 1
            self.last_loud_age += 1
            if voiced:
                self.last_voiced_age = 0
            if db > VAD_DB_ON:
                self.last_loud_age = 0

            vad_active = db > VAD_DB_HARD_OFF and (
                self.last_voiced_age < VAD_VOICED_MEMORY_FR
                or self.last_loud_age < VAD_LOUD_MEMORY_FR
            )

            if voiced and db > SPEECH_UPDATE_FLOOR_DB:
                if not self.speech_db_init:
                    self.speech_db = db
                    self.speech_db_init = True
                else:
                    self.speech_db += SPEECH_LEVEL_ALPHA * (db - self.speech_db)

            if vad_active:
                target_env = max(MIN_ENVELOPE, self._loudness_gain_agc(db))
            else:
                target_env = 0.0
            if target_env > self.envelope:
                self.envelope += ENV_ATTACK * (target_env - self.envelope)
            else:
                self.envelope += ENV_RELEASE * (target_env - self.envelope)
            if not vad_active:
                self.envelope *= MUTE_DECAY

            loud_voiced = self._loudness_gain_agc(db) if voiced else 0.0
            history = list(self.loud_history)
            recent_min = min(history) if history else 0.0

            self.last_nucleus_age += 1
            if (
                vad_active
                and loud_voiced > NUCLEUS_MIN_GAIN
                and (loud_voiced - recent_min) > NUCLEUS_RISE_THRESHOLD
                and self.last_nucleus_age >= NUCLEUS_MIN_SPACING_FR
            ):
                self.target_dir = self._new_target_direction(loud_voiced)
                self.last_nucleus_age = 0

            self.target_dir = (self.target_dir * DIR_DECAY).astype(np.float32)
            self.current_dir = (
                self.current_dir + DIR_LERP * (self.target_dir - self.current_dir)
            ).astype(np.float32)
            if not vad_active:
                self.current_dir *= MUTE_DECAY
                self.target_dir = np.zeros(6, dtype=np.float32)

            # F0 prosody pitch tilt.
            if voiced and f0_hz > 0.0 and vad_active:
                if self.f0_smoothed <= 0.0:
                    self.f0_smoothed = f0_hz
                else:
                    self.f0_smoothed += F0_SMOOTH_ALPHA * (f0_hz - self.f0_smoothed)
                f0_st = 12.0 * math.log2(self.f0_smoothed / F0_REF_HZ)
                if not self.f0_baseline_init:
                    self.f0_baseline_st = f0_st
                    self.f0_baseline_init = True
                else:
                    self.f0_baseline_st += F0_BASELINE_ALPHA * (f0_st - self.f0_baseline_st)
                tilt_st = f0_st - self.f0_baseline_st
                tilt_deg = max(-TILT_MAX_DEG, min(TILT_MAX_DEG, tilt_st * TILT_DEG_PER_SEMITONE))
                tilt_target = math.radians(tilt_deg) * self.envelope
            else:
                tilt_target = 0.0
            self.tilt_current_rad += TILT_LERP * (tilt_target - self.tilt_current_rad)
            if not vad_active:
                self.tilt_current_rad = 0.0

            env = self.envelope if vad_active else 0.0
            d = self.current_dir
            two_pi_t = 2.0 * math.pi * self.t
            breath_pitch = math.radians(BREATH_A_PITCH_DEG) * env * math.sin(BREATH_F_PITCH * two_pi_t + self.phase_pitch)
            breath_yaw = math.radians(BREATH_A_YAW_DEG) * env * math.sin(BREATH_F_YAW * two_pi_t + self.phase_yaw)
            breath_roll = math.radians(BREATH_A_ROLL_DEG) * env * math.sin(BREATH_F_ROLL * two_pi_t + self.phase_roll)
            breath_x = BREATH_A_X_MM * env * math.sin(BREATH_F_X * two_pi_t + self.phase_x)
            breath_y = BREATH_A_Y_MM * env * math.sin(BREATH_F_Y * two_pi_t + self.phase_y)
            breath_z = BREATH_A_Z_MM * env * math.sin(BREATH_F_Z * two_pi_t + self.phase_z)
            pitch = math.radians(A_PITCH_DEG) * float(d[0]) * env + breath_pitch + TILT_SIGN * self.tilt_current_rad
            yaw = math.radians(A_YAW_DEG) * float(d[1]) * env + breath_yaw
            roll = math.radians(A_ROLL_DEG) * float(d[2]) * env + breath_roll
            x_mm = A_X_MM * float(d[3]) * env + breath_x
            y_mm = A_Y_MM * float(d[4]) * env + breath_y
            z_mm = A_Z_MM * float(d[5]) * env + breath_z

            self.loud_history.append(loud_voiced)
            self.t += HOP_MS / 1000.0

            out.append({
                "pitch_rad": pitch,
                "yaw_rad": yaw,
                "roll_rad": roll,
                "x_mm": x_mm,
                "y_mm": y_mm,
                "z_mm": z_mm,
            })

        return out
