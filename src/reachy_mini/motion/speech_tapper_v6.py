"""V6: v5 plus an emotional colouring input.

The v5 signal path is kept as is (strict gate, speaker-relative AGC,
rising-edge nucleus triggers, direction tracking, breath layer, F0-relative
pitch tilt). v6 layers a set of cheap modulations on top, selected by an
``emotion`` name and scaled by an ``energy`` factor. Both can be changed at
runtime with :meth:`SwayRollRT.set_emotion`. Every added term is either
multiplied by the voiced envelope or bounded to a short, self-terminating
gesture, so silences stay silent.

Colourings (all amplitudes at energy 1.0, before the per-axis safety clamp):

| emotion  | what changes on top of v5                                            |
|----------|----------------------------------------------------------------------|
| neutral  | v5 verbatim, plus the two "alive" layers below at modest gains.      |
| angry    | Faster attack (DIR_LERP 0.55, ENV_ATTACK 0.9), sharper decay, pitch  |
|          | and x gains up. Each nucleus adds a downward pitch jab (5 deg) with  |
|          | a forward x thrust (4 mm). Slight constant forward lean (5 mm).      |
| sassy    | Every 3 nuclei the head glides to the other side: yaw +-7 deg and a  |
|          | roll tilt +-5 deg that alternate together. Yaw and roll gains up,    |
|          | breath a touch bigger, chin slightly up.                             |
| sad      | Gains 0.8 to 0.85, breath 0.9, tilt 0.8 (valence, not arousal: use `energy` for that). Slower smoothing (DIR_LERP      |
|          | 0.25, slow envelope). Head biased down 5 deg and sunk 2 mm. On some  |
|          | nuclei (15 %, at most every 2.5 s) a slow extra drop of 4 deg over   |
|          | 1.5 s.                                                               |
| pleading | Head up 4 deg and forward 5 mm, small constant roll (3 deg). Softer  |
|          | gestures (gains 0.95 to 1.0) but more frequent nuclei (spacing 80 ms, |
|          | rise 0.06) and a soft 3 deg nod on each one. Tilt gain 1.3.          |

Two extra layers, active in every colouring:

* Phrase-final tilt: when the gate closes after at least 500 ms of speech,
  the F0 slope over the last 300 ms of voicing decides a short (300 ms)
  half-sine pitch gesture: up on a rising contour (question), down on a
  falling one (statement). It returns to exactly zero, then the silence is
  strictly silent again.
* Phrase-start yaw drift: each time the gate opens, a new small random yaw
  target (+-drift_yaw_deg) is drawn and the head glides to it over ~1 s
  while the phrase lasts. It is envelope-gated and decays between phrases.

``energy`` is clamped to [0, 1.5] and scales every emotional term as well as
the v5 gesture amplitudes. Whatever the colouring, the final per-axis output
is clamped to 1.5 times the largest value v5 can produce on that axis.

Sign conventions assumed (same as v5): positive pitch_rad is nose down,
so "head up" is negative pitch_rad (TILT_SIGN). Positive x_mm is forward.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from itertools import islice

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables shared with v5 (kept identical so neutral == v5 apart from the two
# extra layers)
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

F0_REF_HZ = 100.0
F0_SMOOTH_ALPHA = 0.5
F0_BASELINE_ALPHA = 0.020
TILT_DEG_PER_SEMITONE = 1.5
TILT_MAX_DEG = 8.0
TILT_LERP = 0.40
TILT_SIGN = -1.0

# ---------------------------------------------------------------------------
# v6 additions
# ---------------------------------------------------------------------------
# Multiply "degrees of head up" by this to get the pitch_rad sign.
PITCH_UP = TILT_SIGN
ENERGY_MIN = 0.0
ENERGY_MAX = 1.5
# Per-hop lerp on the continuous emotion parameters (gains, biases, energy)
# so set_emotion() at runtime never steps the head.
PARAM_LERP = 0.15

# Safety clamp: 1.5 x the largest value v5 can produce on each axis.
SAFE_PITCH_RAD = math.radians(1.5 * (A_PITCH_DEG + BREATH_A_PITCH_DEG + TILT_MAX_DEG))
SAFE_YAW_RAD = math.radians(1.5 * (A_YAW_DEG + BREATH_A_YAW_DEG))
SAFE_ROLL_RAD = math.radians(1.5 * (A_ROLL_DEG + BREATH_A_ROLL_DEG))
SAFE_X_MM = 1.5 * (A_X_MM + BREATH_A_X_MM)
SAFE_Y_MM = 1.5 * (A_Y_MM + BREATH_A_Y_MM)
SAFE_Z_MM = 1.5 * (A_Z_MM + BREATH_A_Z_MM)

JAB_DECAY = 0.55                 # per hop, ~2 hops to vanish
NOD_HOPS = 4                     # 200 ms soft nod
GLIDE_LERP = 0.18                # ~280 ms yaw/roll glide
DROP_DOWN_HOPS = 12              # 600 ms going down
DROP_UP_HOPS = 18                # 900 ms coming back
DROP_MIN_SPACING_HOPS = 50       # at most one slow drop per 2.5 s
DRIFT_LERP = 0.06                # ~800 ms glide to the phrase drift target
FINAL_HOPS = 6                   # 300 ms phrase-final gesture
FINAL_MIN_PHRASE_HOPS = 10       # only after 500 ms of speech
FINAL_MIN_SPACING_HOPS = 20      # at most one per second
FINAL_SLOPE_ST = 1.0             # semitones over the last 300 ms to count as rise/fall
F0_HIST_HOPS = 6

# Derived
VAD_VOICED_MEMORY_FR = max(1, int(VAD_VOICED_MEMORY_MS / HOP_MS))
VAD_LOUD_MEMORY_FR = max(1, int(VAD_LOUD_MEMORY_MS / HOP_MS))


@dataclass(frozen=True)
class EmotionProfile:
    """One colouring. Amplitudes in degrees and millimetres at energy 1.0."""

    name: str
    gain: tuple[float, float, float, float, float, float]  # pitch, yaw, roll, x, y, z
    breath_gain: float = 1.0
    tilt_gain: float = 1.0
    dir_lerp: float = DIR_LERP
    dir_decay: float = DIR_DECAY
    env_attack: float = ENV_ATTACK
    env_release: float = ENV_RELEASE
    nucleus_spacing_ms: int = NUCLEUS_MIN_SPACING_MS
    nucleus_rise: float = NUCLEUS_RISE_THRESHOLD
    pitch_bias_up_deg: float = 0.0   # positive = head up
    roll_bias_deg: float = 0.0
    x_bias_mm: float = 0.0           # positive = forward
    z_bias_mm: float = 0.0
    jab_deg: float = 0.0             # per-nucleus downward pitch jab
    jab_x_mm: float = 0.0            # forward thrust riding on the jab
    nod_deg: float = 0.0             # per-nucleus soft nod (down then back)
    glide_yaw_deg: float = 0.0       # side-to-side yaw glide on syllable groups
    glide_roll_deg: float = 0.0      # roll tilt alternating with the glide
    glide_group: int = 3             # nuclei per glide side
    drop_prob: float = 0.0           # chance per nucleus of a slow drop
    drop_deg: float = 0.0
    drift_yaw_deg: float = 3.0       # phrase-start yaw drift range
    final_tilt_deg: float = 2.5      # phrase-final tilt amplitude


PROFILES: dict[str, EmotionProfile] = {
    "neutral": EmotionProfile(
        name="neutral",
        gain=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    ),
    "angry": EmotionProfile(
        name="angry",
        gain=(1.25, 1.0, 0.9, 1.2, 1.0, 1.0),
        breath_gain=0.8,
        tilt_gain=1.2,
        dir_lerp=0.55,
        dir_decay=0.84,
        env_attack=0.9,
        env_release=0.35,
        nucleus_spacing_ms=100,
        nucleus_rise=0.07,
        pitch_bias_up_deg=-1.0,
        x_bias_mm=5.0,
        jab_deg=5.0,
        jab_x_mm=4.0,
        drift_yaw_deg=2.0,
        final_tilt_deg=3.5,
    ),
    "sassy": EmotionProfile(
        name="sassy",
        gain=(0.9, 1.1, 1.2, 0.9, 1.1, 1.0),
        breath_gain=1.1,
        tilt_gain=1.1,
        pitch_bias_up_deg=1.0,
        z_bias_mm=1.0,
        glide_yaw_deg=7.0,
        glide_roll_deg=5.0,
        glide_group=3,
        drift_yaw_deg=4.0,
        final_tilt_deg=3.0,
    ),
    "sad": EmotionProfile(
        name="sad",
        gain=(0.85, 0.8, 0.85, 0.85, 0.8, 0.85),
        breath_gain=0.9,
        tilt_gain=0.8,
        dir_lerp=0.25,
        dir_decay=0.92,
        env_attack=0.4,
        env_release=0.2,
        nucleus_spacing_ms=140,
        nucleus_rise=0.10,
        pitch_bias_up_deg=-5.0,
        roll_bias_deg=2.0,
        x_bias_mm=-2.0,
        z_bias_mm=-2.0,
        drop_prob=0.15,
        drop_deg=4.0,
        drift_yaw_deg=1.5,
        final_tilt_deg=3.0,
    ),
    "pleading": EmotionProfile(
        name="pleading",
        gain=(0.95, 0.95, 1.0, 1.0, 1.0, 1.0),
        breath_gain=1.0,
        tilt_gain=1.3,
        dir_lerp=0.45,
        nucleus_spacing_ms=80,
        nucleus_rise=0.06,
        pitch_bias_up_deg=4.0,
        roll_bias_deg=3.0,
        x_bias_mm=5.0,
        z_bias_mm=1.0,
        nod_deg=3.0,
        drift_yaw_deg=2.0,
        final_tilt_deg=2.5,
    ),
}

EMOTIONS: tuple[str, ...] = tuple(PROFILES)


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


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def resolve_profile(emotion: str | None) -> EmotionProfile:
    """Map an emotion name to a profile; unknown names fall back to neutral."""
    key = (emotion or "neutral").strip().lower()
    prof = PROFILES.get(key)
    if prof is None:
        logger.warning("Unknown wobbler emotion %r, using neutral", emotion)
        prof = PROFILES["neutral"]
    return prof


class SwayRollRT:
    """V6: v5 with emotional colouring and two phrase-level layers."""

    def __init__(
        self,
        rng_seed: int = 7,
        sample_rate: int = 16_000,
        emotion: str = "neutral",
        energy: float = 1.0,
    ) -> None:
        self._seed = int(rng_seed)
        self.sample_rate = int(sample_rate)
        self.frame = int(self.sample_rate * FRAME_MS / 1000)
        self.hop = int(self.sample_rate * HOP_MS / 1000)
        self.samples: deque[float] = deque(maxlen=10 * self.sample_rate)
        self.carry: NDArray[np.float32] = np.zeros(0, dtype=np.float32)

        self.rng = np.random.default_rng(self._seed)
        self.phase_pitch = float(self.rng.random() * 2 * math.pi)
        self.phase_yaw = float(self.rng.random() * 2 * math.pi)
        self.phase_roll = float(self.rng.random() * 2 * math.pi)
        self.phase_x = float(self.rng.random() * 2 * math.pi)
        self.phase_y = float(self.rng.random() * 2 * math.pi)
        self.phase_z = float(self.rng.random() * 2 * math.pi)

        self.profile = resolve_profile(emotion)
        self.energy = _clamp(float(energy), ENERGY_MIN, ENERGY_MAX)
        # Continuous parameters start at their targets (no fade-in at construction).
        self._params = self._target_params()
        self._reset_state()

    # ------------------------------------------------------------------
    # Emotion handling
    # ------------------------------------------------------------------
    def set_emotion(self, emotion: str, energy: float | None = None) -> None:
        """Switch colouring at runtime. Continuous terms cross-fade over ~300 ms."""
        self.profile = resolve_profile(emotion)
        if energy is not None:
            self.energy = _clamp(float(energy), ENERGY_MIN, ENERGY_MAX)

    def _target_params(self) -> NDArray[np.float64]:
        p = self.profile
        return np.array(
            [
                *p.gain,
                p.breath_gain,
                p.tilt_gain,
                p.pitch_bias_up_deg,
                p.roll_bias_deg,
                p.x_bias_mm,
                p.z_bias_mm,
                self.energy,
            ],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def _reset_state(self) -> None:
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
        self.t = 0.0
        # v6 state
        self.prev_vad = False
        self.phrase_hops = 0
        self._phrase_len_at_end = 0
        self.nucleus_count = 0
        self.jab_current = 0.0
        self.nod_phase = NOD_HOPS
        self.glide_side = 1.0
        self.glide_yaw_current = 0.0
        self.glide_roll_current = 0.0
        self.drop_phase = DROP_DOWN_HOPS + DROP_UP_HOPS
        self.last_drop_age = 999
        self.drift_target_deg = 0.0
        self.drift_current_deg = 0.0
        self.f0_st_hist: deque[float] = deque(maxlen=F0_HIST_HOPS)
        self.final_phase = FINAL_HOPS
        self.final_amp_deg = 0.0
        self.last_final_age = 999

    def reset(self) -> None:
        self.samples.clear()
        self.carry = np.zeros(0, dtype=np.float32)
        self._params = self._target_params()
        self._reset_state()

    # ------------------------------------------------------------------
    # Helpers shared with v5
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
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

            prof = self.profile
            # Cross-fade the continuous emotion parameters.
            self._params += PARAM_LERP * (self._target_params() - self._params)
            gains = self._params[0:6]
            breath_gain = float(self._params[6])
            tilt_gain = float(self._params[7])
            pitch_bias_up_deg = float(self._params[8])
            roll_bias_deg = float(self._params[9])
            x_bias_mm = float(self._params[10])
            z_bias_mm = float(self._params[11])
            energy = float(self._params[12])

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
            phrase_start = vad_active and not self.prev_vad
            phrase_end = self.prev_vad and not vad_active
            self.prev_vad = vad_active
            if phrase_end:
                self._phrase_len_at_end = self.phrase_hops
            self.phrase_hops = self.phrase_hops + 1 if vad_active else 0

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
                self.envelope += prof.env_attack * (target_env - self.envelope)
            else:
                self.envelope += prof.env_release * (target_env - self.envelope)
            if not vad_active:
                self.envelope *= MUTE_DECAY

            loud_voiced = self._loudness_gain_agc(db) if voiced else 0.0
            history = list(self.loud_history)
            recent_min = min(history) if history else 0.0

            spacing_fr = max(1, int(prof.nucleus_spacing_ms / HOP_MS))
            self.last_nucleus_age += 1
            nucleus = (
                vad_active
                and loud_voiced > NUCLEUS_MIN_GAIN
                and (loud_voiced - recent_min) > prof.nucleus_rise
                and self.last_nucleus_age >= spacing_fr
            )
            if nucleus:
                self.target_dir = self._new_target_direction(loud_voiced)
                self.last_nucleus_age = 0
                self._on_nucleus(loud_voiced)

            self.target_dir = (self.target_dir * prof.dir_decay).astype(np.float32)
            self.current_dir = (
                self.current_dir + prof.dir_lerp * (self.target_dir - self.current_dir)
            ).astype(np.float32)
            if not vad_active:
                self.current_dir *= MUTE_DECAY
                self.target_dir = np.zeros(6, dtype=np.float32)

            # F0 prosody pitch tilt (v5), scaled by the colouring's tilt gain.
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
                self.f0_st_hist.append(f0_st)
                tilt_st = f0_st - self.f0_baseline_st
                tilt_deg = _clamp(tilt_st * TILT_DEG_PER_SEMITONE, -TILT_MAX_DEG, TILT_MAX_DEG)
                tilt_target = math.radians(tilt_deg) * self.envelope * tilt_gain
            else:
                tilt_target = 0.0
            self.tilt_current_rad += TILT_LERP * (tilt_target - self.tilt_current_rad)
            if not vad_active:
                self.tilt_current_rad = 0.0

            # Phrase-level layers.
            if phrase_start:
                self.drift_target_deg = float(self.rng.uniform(-1.0, 1.0)) * prof.drift_yaw_deg
                self.f0_st_hist.clear()
            if phrase_end:
                self.drift_target_deg = 0.0
                self._maybe_start_final_tilt()
            self.drift_current_deg += DRIFT_LERP * (self.drift_target_deg - self.drift_current_deg)
            if not vad_active:
                self.drift_current_deg *= MUTE_DECAY

            # Per-hop progress of the short gestures.
            self.jab_current *= JAB_DECAY
            self.nod_phase += 1
            self.drop_phase += 1
            self.last_drop_age += 1
            self.final_phase += 1
            self.last_final_age += 1
            self.glide_yaw_current += GLIDE_LERP * (
                self.glide_side * prof.glide_yaw_deg - self.glide_yaw_current
            )
            self.glide_roll_current += GLIDE_LERP * (
                self.glide_side * prof.glide_roll_deg - self.glide_roll_current
            )
            if not vad_active:
                self.jab_current = 0.0
                self.nod_phase = NOD_HOPS
                self.drop_phase = DROP_DOWN_HOPS + DROP_UP_HOPS

            env = self.envelope if vad_active else 0.0
            d = self.current_dir
            two_pi_t = 2.0 * math.pi * self.t
            b = breath_gain * energy * env
            breath_pitch = math.radians(BREATH_A_PITCH_DEG) * b * math.sin(BREATH_F_PITCH * two_pi_t + self.phase_pitch)
            breath_yaw = math.radians(BREATH_A_YAW_DEG) * b * math.sin(BREATH_F_YAW * two_pi_t + self.phase_yaw)
            breath_roll = math.radians(BREATH_A_ROLL_DEG) * b * math.sin(BREATH_F_ROLL * two_pi_t + self.phase_roll)
            breath_x = BREATH_A_X_MM * b * math.sin(BREATH_F_X * two_pi_t + self.phase_x)
            breath_y = BREATH_A_Y_MM * b * math.sin(BREATH_F_Y * two_pi_t + self.phase_y)
            breath_z = BREATH_A_Z_MM * b * math.sin(BREATH_F_Z * two_pi_t + self.phase_z)

            g = env * energy
            pitch = math.radians(A_PITCH_DEG) * float(gains[0]) * float(d[0]) * g + breath_pitch
            yaw = math.radians(A_YAW_DEG) * float(gains[1]) * float(d[1]) * g + breath_yaw
            roll = math.radians(A_ROLL_DEG) * float(gains[2]) * float(d[2]) * g + breath_roll
            x_mm = A_X_MM * float(gains[3]) * float(d[3]) * g + breath_x
            y_mm = A_Y_MM * float(gains[4]) * float(d[4]) * g + breath_y
            z_mm = A_Z_MM * float(gains[5]) * float(d[5]) * g + breath_z

            # F0 tilt (already envelope-gated) scaled by energy.
            pitch += TILT_SIGN * self.tilt_current_rad * energy

            # Emotional biases and gestures, all envelope-gated.
            pitch += PITCH_UP * math.radians(pitch_bias_up_deg) * g
            roll += math.radians(roll_bias_deg) * g
            x_mm += x_bias_mm * g
            z_mm += z_bias_mm * g

            if self.jab_current > 1e-4:
                pitch += -PITCH_UP * math.radians(self.jab_current) * g
                if prof.jab_deg > 0.0:
                    x_mm += prof.jab_x_mm * (self.jab_current / prof.jab_deg) * g
            if self.nod_phase < NOD_HOPS:
                nod = prof.nod_deg * math.sin(math.pi * self.nod_phase / NOD_HOPS)
                pitch += -PITCH_UP * math.radians(nod) * g
            drop_total = DROP_DOWN_HOPS + DROP_UP_HOPS
            if self.drop_phase < drop_total:
                if self.drop_phase < DROP_DOWN_HOPS:
                    shape = math.sin(0.5 * math.pi * self.drop_phase / DROP_DOWN_HOPS)
                else:
                    shape = math.cos(0.5 * math.pi * (self.drop_phase - DROP_DOWN_HOPS) / DROP_UP_HOPS)
                pitch += -PITCH_UP * math.radians(prof.drop_deg * shape) * g
            yaw += math.radians(self.glide_yaw_current) * g
            roll += math.radians(self.glide_roll_current) * g
            yaw += math.radians(self.drift_current_deg) * g

            # Phrase-final tilt: the one term that runs into the silence. It is
            # a 300 ms half-sine that returns to exactly zero.
            if self.final_phase < FINAL_HOPS:
                shape = math.sin(math.pi * self.final_phase / FINAL_HOPS)
                pitch += PITCH_UP * math.radians(self.final_amp_deg * shape) * energy

            pitch = _clamp(pitch, -SAFE_PITCH_RAD, SAFE_PITCH_RAD)
            yaw = _clamp(yaw, -SAFE_YAW_RAD, SAFE_YAW_RAD)
            roll = _clamp(roll, -SAFE_ROLL_RAD, SAFE_ROLL_RAD)
            x_mm = _clamp(x_mm, -SAFE_X_MM, SAFE_X_MM)
            y_mm = _clamp(y_mm, -SAFE_Y_MM, SAFE_Y_MM)
            z_mm = _clamp(z_mm, -SAFE_Z_MM, SAFE_Z_MM)

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

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------
    def _on_nucleus(self, prominence: float) -> None:
        prof = self.profile
        self.nucleus_count += 1
        if prof.jab_deg > 0.0:
            self.jab_current = prof.jab_deg * (0.6 + 0.4 * prominence)
        if prof.nod_deg > 0.0:
            self.nod_phase = 0
        if prof.glide_yaw_deg > 0.0 or prof.glide_roll_deg > 0.0:
            if self.nucleus_count % max(1, prof.glide_group) == 0:
                self.glide_side = -self.glide_side
        if (
            prof.drop_prob > 0.0
            and self.last_drop_age >= DROP_MIN_SPACING_HOPS
            and float(self.rng.random()) < prof.drop_prob
        ):
            self.drop_phase = 0
            self.last_drop_age = 0

    def _maybe_start_final_tilt(self) -> None:
        """Called when the gate closes. Decide the phrase-final gesture."""
        prof = self.profile
        hist = list(self.f0_st_hist)
        if (
            prof.final_tilt_deg <= 0.0
            or self.last_final_age < FINAL_MIN_SPACING_HOPS
            or len(hist) < 4
        ):
            return
        if self._phrase_len_at_end < FINAL_MIN_PHRASE_HOPS:
            return
        k = len(hist) // 2
        slope = float(np.mean(hist[-k:]) - np.mean(hist[:k]))
        if abs(slope) < FINAL_SLOPE_ST:
            return
        sign = 1.0 if slope > 0.0 else -1.0
        self.final_amp_deg = sign * prof.final_tilt_deg * _clamp(abs(slope) / 2.0, 0.5, 1.0)
        self.final_phase = 0
        self.last_final_age = 0
