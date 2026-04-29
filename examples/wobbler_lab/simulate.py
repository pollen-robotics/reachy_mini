"""Run a speech_tapper offline against a WAV, capture per-hop motion."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType

import librosa
import numpy as np
from numpy.typing import NDArray

VERSIONS = {
    "v0": "reachy_mini.motion.speech_tapper",
    "v1": "reachy_mini.motion.speech_tapper_v1",
    "v2": "reachy_mini.motion.speech_tapper_v2",
    "v3": "reachy_mini.motion.speech_tapper_v3",
    "v4": "reachy_mini.motion.speech_tapper_v4",
    "v5": "reachy_mini.motion.speech_tapper_v5",
}


@dataclass
class SimResult:
    audio: NDArray[np.float32]
    sample_rate: int
    motion: NDArray[np.float32]            # (T, 6) — pitch, yaw, roll (rad), x, y, z (mm)
    motion_time: NDArray[np.float32]       # (T,)
    hop_ms: int


def _load_module(version: str) -> ModuleType:
    name = VERSIONS[version]
    return importlib.import_module(name)


def load_audio(path: str, target_sr: int = 16_000) -> tuple[NDArray[np.float32], int]:
    """Load WAV → mono float32 at *target_sr*."""
    pcm, sr = librosa.load(path, sr=target_sr, mono=True)
    return pcm.astype(np.float32), int(sr)


def run_tapper(version: str, pcm: NDArray[np.float32], sample_rate: int) -> SimResult:
    """Feed *pcm* through *version*'s SwayRollRT in 1-s chunks, collect motion."""
    mod = _load_module(version)
    sway = mod.SwayRollRT(sample_rate=sample_rate)
    chunk = sample_rate
    results: list[dict[str, float]] = []
    for i in range(0, len(pcm), chunk):
        results.extend(sway.feed(pcm[i:i + chunk]))

    if not results:
        motion = np.zeros((0, 6), dtype=np.float32)
    else:
        motion = np.array(
            [
                [r["pitch_rad"], r["yaw_rad"], r["roll_rad"], r["x_mm"], r["y_mm"], r["z_mm"]]
                for r in results
            ],
            dtype=np.float32,
        )

    hop_ms = int(mod.HOP_MS)
    times = np.arange(len(motion), dtype=np.float32) * (hop_ms / 1000.0)
    return SimResult(
        audio=pcm,
        sample_rate=sample_rate,
        motion=motion,
        motion_time=times,
        hop_ms=hop_ms,
    )
