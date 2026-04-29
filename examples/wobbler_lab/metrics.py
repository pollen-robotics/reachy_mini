"""Synchronization metrics between motion and speech features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SyncMetrics:
    stillness_in_silence: float       # 1 - mean|m| in silent frames / mean|m| overall (higher = better)
    voiced_unvoiced_ratio: float      # mean|m| voiced / mean|m| unvoiced (higher = better)
    onset_alignment: float            # peak-velocity-near-onset / random-shift baseline (>1 = better)


def motion_magnitude(motion: NDArray[np.float32]) -> NDArray[np.float32]:
    """Per-axis-normalized sum of |motion|."""
    if motion.size == 0:
        return np.zeros(0, dtype=np.float32)
    norm = np.std(motion, axis=0)
    norm = np.where(norm < 1e-9, 1.0, norm)
    return np.sum(np.abs(motion / norm), axis=1).astype(np.float32)


def motion_velocity(motion: NDArray[np.float32]) -> NDArray[np.float32]:
    """Per-axis-normalized |Δmotion|."""
    if len(motion) < 2:
        return np.zeros(len(motion), dtype=np.float32)
    diff = np.diff(motion, axis=0, prepend=motion[:1])
    norm = np.std(diff, axis=0) + 1e-9
    return np.sum(np.abs(diff / norm), axis=1).astype(np.float32)


def compute(
    motion: NDArray[np.float32],
    voiced: NDArray[np.bool_],
    rms_db: NDArray[np.float32],
    onset_idx: NDArray[np.int_],
    hop_ms: int,
    silence_thresh_db: float = -55.0,
    onset_window_ms: int = 150,
    n_baseline: int = 25,
) -> SyncMetrics:
    """Score how well *motion* tracks the speech features."""
    n = min(len(motion), len(voiced), len(rms_db))
    if n == 0:
        return SyncMetrics(0.0, 0.0, 0.0)
    motion = motion[:n]
    voiced = voiced[:n]
    rms_db = rms_db[:n]

    mag = motion_magnitude(motion)
    silent = rms_db < silence_thresh_db
    overall = float(np.mean(mag) + 1e-9)
    sis = 1.0 - float(np.mean(mag[silent]) / overall) if silent.any() else 0.0
    sis = max(-2.0, min(2.0, sis))  # clamp to a sane range for printing

    voiced_mean = float(np.mean(mag[voiced])) if voiced.any() else 0.0
    unvoiced_mean = float(np.mean(mag[~voiced])) if (~voiced).any() else 1e-9
    vur = voiced_mean / max(unvoiced_mean, 1e-9)

    vel = motion_velocity(motion)
    win = max(1, int(onset_window_ms / hop_ms))
    keep = onset_idx[(onset_idx >= 0) & (onset_idx < n - win)]
    if len(keep) == 0 or vel.max() < 1e-9:
        oa = 0.0
    else:
        true_score = float(np.mean([np.max(vel[i:i + win]) for i in keep]))
        rng = np.random.default_rng(0)
        baseline = []
        for _ in range(n_baseline):
            shift = int(rng.integers(low=win, high=max(win + 1, n - win)))
            shifted = (keep + shift) % (n - win)
            baseline.append(float(np.mean([np.max(vel[i:i + win]) for i in shifted])))
        denom = float(np.mean(baseline))
        oa = true_score / max(denom, 1e-9)

    return SyncMetrics(
        stillness_in_silence=float(sis),
        voiced_unvoiced_ratio=float(vur),
        onset_alignment=float(oa),
    )
