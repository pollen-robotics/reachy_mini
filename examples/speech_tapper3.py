# sway_rt.py
import math
from collections import deque
from itertools import islice
from typing import List, Dict, Optional
import numpy as np

SR = 16_000
FRAME_MS = 20
HOP_MS = 10

SWAY_MASTER = 1.5
SENS_DB_OFFSET = +4.0
VAD_DB_ON = -35.0
VAD_DB_OFF = -45.0
VAD_ATTACK_MS = 40
VAD_RELEASE_MS = 250
ENV_FOLLOW_GAIN = 0.65

SWAY_F_PITCH = 2.2;  SWAY_A_PITCH_DEG = 4.5
SWAY_F_YAW   = 0.6;  SWAY_A_YAW_DEG   = 7.5
SWAY_F_ROLL  = 1.3;  SWAY_A_ROLL_DEG  = 2.25
SWAY_F_X     = 0.35; SWAY_A_X_MM      = 4.5
SWAY_F_Y     = 0.45; SWAY_A_Y_MM      = 3.75
SWAY_F_Z     = 0.25; SWAY_A_Z_MM      = 2.25

SWAY_DB_LOW = -46.0
SWAY_DB_HIGH = -18.0
LOUDNESS_GAMMA = 0.9
SWAY_ATTACK_MS = 50
SWAY_RELEASE_MS = 250

FRAME = int(SR * FRAME_MS / 1000)
HOP   = int(SR * HOP_MS  / 1000)
ATTACK_FR       = max(1, int(VAD_ATTACK_MS   / HOP_MS))
RELEASE_FR      = max(1, int(VAD_RELEASE_MS  / HOP_MS))
SWAY_ATTACK_FR  = max(1, int(SWAY_ATTACK_MS  / HOP_MS))
SWAY_RELEASE_FR = max(1, int(SWAY_RELEASE_MS / HOP_MS))

def _rms_dbfs(x: np.ndarray) -> float:
    x = x.astype(np.float32, copy=False)
    rms = np.sqrt(np.mean(x * x) + 1e-12)
    return 20.0 * math.log10(rms + 1e-12)

def _loudness_gain(db: float, offset: float = SENS_DB_OFFSET) -> float:
    t = (db + offset - SWAY_DB_LOW) / (SWAY_DB_HIGH - SWAY_DB_LOW)
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    return t ** LOUDNESS_GAMMA if LOUDNESS_GAMMA != 1.0 else t

def _to_float32_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = np.mean(x, axis=0 if x.shape[0] < x.shape[1] else 1)
    if not np.issubdtype(x.dtype, np.floating):
        info = np.iinfo(x.dtype)
        x = x.astype(np.float32) / max(-info.min, info.max)
    else:
        x = x.astype(np.float32, copy=False)
    return x

def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or x.size == 0:
        return x
    t_in = np.arange(x.size) / sr_in
    t_out = np.arange(int(round(x.size * sr_out / sr_in))) / sr_out
    return np.interp(t_out, t_in, x).astype(np.float32)

class SwayRollRT:
    """Feed audio chunks → per-hop sway outputs."""
    def __init__(self, rng_seed: int = 7):
        self.samples = deque(maxlen=10 * SR)
        self.carry = np.zeros(0, dtype=np.float32)
        self.frame_idx = 0
        self.vad_on = False; self.vad_above = 0; self.vad_below = 0
        self.sway_env = 0.0; self.sway_up = 0; self.sway_down = 0
        rng = np.random.default_rng(rng_seed)
        self.phase_pitch = rng.random() * 2 * math.pi
        self.phase_yaw   = rng.random() * 2 * math.pi
        self.phase_roll  = rng.random() * 2 * math.pi
        self.phase_x     = rng.random() * 2 * math.pi
        self.phase_y     = rng.random() * 2 * math.pi
        self.phase_z     = rng.random() * 2 * math.pi
        self.t = 0.0

    def reset(self): self.__init__()

    def feed(self, pcm: np.ndarray, sr: Optional[int]) -> List[Dict[str, float]]:
        sr_in = SR if sr is None else int(sr)
        x = _to_float32_mono(np.asarray(pcm))
        if sr_in != SR:
            x = _resample_linear(x, sr_in, SR)
        if x.size == 0:
            return []

        self.carry = np.concatenate([self.carry, x])
        out: List[Dict[str, float]] = []

        while self.carry.size >= HOP:
            hop = self.carry[:HOP]; self.carry = self.carry[HOP:]
            self.samples.extend(hop.tolist())
            if len(self.samples) < FRAME:
                self.t += HOP_MS / 1000.0; self.frame_idx += 1; continue

            frame = np.fromiter(
                islice(self.samples, len(self.samples)-FRAME, len(self.samples)),
                dtype=np.float32, count=FRAME
            )
            db = _rms_dbfs(frame)

            if db >= VAD_DB_ON:
                self.vad_above += 1; self.vad_below = 0
                if not self.vad_on and self.vad_above >= ATTACK_FR: self.vad_on = True
            elif db <= VAD_DB_OFF:
                self.vad_below += 1; self.vad_above = 0
                if self.vad_on and self.vad_below >= RELEASE_FR: self.vad_on = False

            if self.vad_on:
                self.sway_up   = min(SWAY_ATTACK_FR,  self.sway_up + 1); self.sway_down = 0
            else:
                self.sway_down = min(SWAY_RELEASE_FR, self.sway_down + 1); self.sway_up = 0
            up = self.sway_up / SWAY_ATTACK_FR
            down = 1.0 - (self.sway_down / SWAY_RELEASE_FR)
            target = up if self.vad_on else down
            self.sway_env += ENV_FOLLOW_GAIN * (target - self.sway_env)
            self.sway_env = 0.0 if self.sway_env < 0 else (1.0 if self.sway_env > 1 else self.sway_env)

            loud = _loudness_gain(db) * SWAY_MASTER
            env = self.sway_env
            self.t += HOP_MS / 1000.0

            pitch = math.radians(SWAY_A_PITCH_DEG) * loud * env * math.sin(2*math.pi*SWAY_F_PITCH*self.t + self.phase_pitch)
            yaw   = math.radians(SWAY_A_YAW_DEG)   * loud * env * math.sin(2*math.pi*SWAY_F_YAW  *self.t + self.phase_yaw)
            roll  = math.radians(SWAY_A_ROLL_DEG)  * loud * env * math.sin(2*math.pi*SWAY_F_ROLL *self.t + self.phase_roll)
            x_mm  = SWAY_A_X_MM * loud * env * math.sin(2*math.pi*SWAY_F_X*self.t + self.phase_x)
            y_mm  = SWAY_A_Y_MM * loud * env * math.sin(2*math.pi*SWAY_F_Y*self.t + self.phase_y)
            z_mm  = SWAY_A_Z_MM * loud * env * math.sin(2*math.pi*SWAY_F_Z*self.t + self.phase_z)

            out.append({
                "pitch_rad": pitch, "yaw_rad": yaw, "roll_rad": roll,
                "pitch_deg": math.degrees(pitch), "yaw_deg": math.degrees(yaw), "roll_deg": math.degrees(roll),
                "x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm,
            })
        return out
