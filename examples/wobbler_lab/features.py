"""Audio features (RMS, voicing/F0, onsets, syllable nuclei) at the wobbler hop grid."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks


@dataclass
class AudioFeatures:
    rms_db: NDArray[np.float32]            # (T,) RMS in dBFS at hop grid
    times: NDArray[np.float32]              # (T,) timestamps in seconds
    onset_strength: NDArray[np.float32]    # (T,) librosa.onset_strength
    voiced: NDArray[np.bool_]               # (T,)
    f0: NDArray[np.float32]                 # (T,) NaN where unvoiced
    nucleus_idx: NDArray[np.int_]           # syllable nucleus indices into times
    onset_idx: NDArray[np.int_]             # onset indices into times
    hop_ms: int


def _align(arr: NDArray, target_len: int, pad_value: float = 0.0) -> NDArray:
    """Pad/truncate *arr* to exactly *target_len* samples."""
    if len(arr) > target_len:
        return arr[:target_len]
    if len(arr) < target_len:
        pad = target_len - len(arr)
        return np.pad(arr, (0, pad), constant_values=pad_value)
    return arr


def extract(
    pcm: NDArray[np.float32],
    sample_rate: int,
    hop_ms: int = 50,
    frame_ms: int = 20,
) -> AudioFeatures:
    """Compute hop-grid-aligned audio features for *pcm*."""
    hop_length = int(sample_rate * hop_ms / 1000)
    frame_length = max(int(sample_rate * frame_ms / 1000), hop_length * 2)

    rms = librosa.feature.rms(y=pcm, frame_length=frame_length, hop_length=hop_length).flatten()
    rms_db = 20.0 * np.log10(rms + 1e-10)
    n_frames = len(rms_db)
    times = np.arange(n_frames, dtype=np.float32) * (hop_ms / 1000.0)

    onset_env = librosa.onset.onset_strength(y=pcm, sr=sample_rate, hop_length=hop_length)
    onset_env = _align(onset_env, n_frames, 0.0).astype(np.float32)

    f0, voiced_flag, _ = librosa.pyin(
        pcm,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C6")),
        sr=sample_rate,
        frame_length=frame_length * 4,  # pyin needs longer windows for low pitches
        hop_length=hop_length,
    )
    f0 = _align(np.asarray(f0, dtype=np.float64), n_frames, np.nan).astype(np.float32)
    voiced_flag = _align(np.asarray(voiced_flag, dtype=bool), n_frames, False)

    onset_idx_arr = librosa.onset.onset_detect(
        y=pcm,
        sr=sample_rate,
        hop_length=hop_length,
        backtrack=False,
    )
    onset_idx = np.asarray(onset_idx_arr, dtype=int)

    voiced_db = np.where(voiced_flag, rms_db, -120.0)
    min_distance = max(1, int(0.150 * 1000 / hop_ms))
    peaks, _ = find_peaks(voiced_db, distance=min_distance, prominence=3.0)
    nucleus_idx = np.asarray(peaks, dtype=int)

    return AudioFeatures(
        rms_db=rms_db.astype(np.float32),
        times=times,
        onset_strength=onset_env,
        voiced=voiced_flag,
        f0=f0,
        nucleus_idx=nucleus_idx,
        onset_idx=onset_idx,
        hop_ms=hop_ms,
    )
