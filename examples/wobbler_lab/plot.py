"""Stacked-timeline figure: spectrogram + features + 6-axis motion."""

from __future__ import annotations

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from .features import AudioFeatures
from .simulate import SimResult

AXIS_LABELS = ["pitch (rad)", "yaw (rad)", "roll (rad)", "x (mm)", "y (mm)", "z (mm)"]


def plot(
    sim: SimResult,
    feat: AudioFeatures,
    version: str,
    clip: str,
    out_path: Path,
) -> None:
    """Save a stacked figure aligning audio features and 6-axis motion."""
    n_rows = 10  # spectro, RMS+voicing, F0, onset, pitch, yaw, roll, x, y, z
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(15, 16),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2, 1, 1, 1, 1, 1, 1, 1, 1]},
    )

    pcm = sim.audio
    sr = sim.sample_rate
    hop_length = int(sr * sim.hop_ms / 1000)
    spec = librosa.amplitude_to_db(
        np.abs(librosa.stft(pcm, n_fft=2048, hop_length=hop_length)),
        ref=np.max,
    )
    librosa.display.specshow(
        spec, sr=sr, x_axis="time", y_axis="hz",
        ax=axes[0], hop_length=hop_length, cmap="magma",
    )
    axes[0].set_ylim(0, 5000)
    axes[0].set_ylabel("Hz")
    axes[0].set_title(f"{clip} | {version}")

    times = feat.times
    n = len(times)

    rms_clipped = np.clip(feat.rms_db, -80, 0)
    axes[1].plot(times, rms_clipped, color="black", lw=0.8)
    voiced_mask = np.asarray(feat.voiced[:n], dtype=bool)
    axes[1].fill_between(
        times, -80, rms_clipped,
        where=voiced_mask, color="cornflowerblue", alpha=0.25, label="voiced",
    )
    for idx in feat.nucleus_idx:
        if idx < n:
            axes[1].axvline(times[idx], color="red", lw=0.6, alpha=0.6)
    for idx in feat.onset_idx:
        if idx < n:
            axes[1].axvline(times[idx], color="green", lw=0.4, alpha=0.4, linestyle="--")
    axes[1].set_ylim(-80, 0)
    axes[1].set_ylabel("RMS dB")
    axes[1].legend(loc="upper right", fontsize=7)

    axes[2].plot(times, feat.f0, color="purple", lw=0.8)
    axes[2].set_ylabel("F0 (Hz)")
    axes[2].set_yscale("log")
    axes[2].set_ylim(50, 800)

    axes[3].plot(times, feat.onset_strength, color="darkgreen", lw=0.7)
    axes[3].set_ylabel("onset")
    for idx in feat.onset_idx:
        if idx < n:
            axes[3].axvline(times[idx], color="green", lw=0.4, alpha=0.5, linestyle="--")

    motion = sim.motion
    motion_time = sim.motion_time
    m = min(len(motion), len(motion_time))
    motion = motion[:m]
    motion_time = motion_time[:m]
    for k in range(6):
        ax = axes[4 + k]
        ax.plot(motion_time, motion[:, k], lw=0.7, color="C0")
        ax.set_ylabel(AXIS_LABELS[k], fontsize=8)
        ax.axhline(0, color="black", lw=0.3, alpha=0.3)
        for idx in feat.nucleus_idx:
            if idx < n:
                ax.axvline(times[idx], color="red", lw=0.25, alpha=0.25)

    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
