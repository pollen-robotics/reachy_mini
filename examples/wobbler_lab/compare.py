"""Compare wobbler versions on the lines of a scene: stacked plots + metrics.

Run from the repo root::

    python examples/wobbler_lab/compare.py \
        --wav-dir ../agentic_robot_theater/scenes/couple_fight/audio \
        --scene ../agentic_robot_theater/scenes/couple_fight/scene.json \
        --versions v0,v5,v6 --out-dir examples/wobbler_lab/out/couple_fight

Writes one PNG per wav (spectrogram, RMS + voicing, then rotation and
translation rows for every version), a ``summary.png`` with the metrics per
line and version, and ``metrics.md``. Versions that accept a colouring get
the beat's ``emotion`` / ``energy`` (lookup by wav stem in the scene).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.wobbler_lab import features as feature_mod  # noqa: E402
from examples.wobbler_lab import metrics as metric_mod  # noqa: E402
from examples.wobbler_lab import simulate as sim_mod  # noqa: E402
from examples.wobbler_lab.offsets import colouring_for, load_scene_beats  # noqa: E402

# Fixed categorical order (validated palette): slot 1 blue, 2 orange, 3 aqua.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#2b2b2b"
INK_MUTED = "#7a7a76"
GRID = "#e4e4e0"
SURFACE = "#fcfcfb"


@dataclass
class LineResult:
    stem: str
    version: str
    emotion: str
    energy: float
    stillness: float
    vu_ratio: float
    onset_alignment: float
    peak_rot_deg: float
    peak_trans_mm: float
    rms_rot_deg: float


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.yaxis.label.set_color(INK)
    ax.grid(True, axis="y", color=GRID, lw=0.6)


def plot_line(
    stem: str,
    pcm: np.ndarray,
    sr: int,
    feat: feature_mod.AudioFeatures,
    sims: dict[str, sim_mod.SimResult],
    labels: dict[str, str],
    out_path: Path,
) -> None:
    versions = list(sims)
    n_rows = 2 + 2 * len(versions)
    ratios = [2.2, 1.0] + [1.0, 1.0] * len(versions)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(14, 2.0 + 1.6 * n_rows), sharex=True,
        gridspec_kw={"height_ratios": ratios},
    )
    fig.patch.set_facecolor(SURFACE)

    hop_length = int(sr * 0.05)
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(pcm, n_fft=1024, hop_length=hop_length)), ref=np.max)
    librosa.display.specshow(spec, sr=sr, x_axis="time", y_axis="hz", ax=axes[0], hop_length=hop_length, cmap="magma")
    axes[0].set_ylim(0, 4000)
    axes[0].set_ylabel("Hz")
    axes[0].set_title(f"{stem}   |   " + "   ".join(labels[v] for v in versions), color=INK, fontsize=11, loc="left")

    times = feat.times
    n = len(times)
    rms = np.clip(feat.rms_db, -80, 0)
    axes[1].plot(times, rms, color=INK, lw=0.8)
    axes[1].fill_between(times, -80, rms, where=np.asarray(feat.voiced[:n], dtype=bool), color=SERIES[0], alpha=0.18, label="voiced")
    for idx in feat.nucleus_idx:
        if idx < n:
            axes[1].axvline(times[idx], color=SERIES[1], lw=0.6, alpha=0.7)
    axes[1].plot([], [], color=SERIES[1], lw=0.6, label="syllable nucleus")
    axes[1].set_ylim(-80, 0)
    axes[1].set_ylabel("RMS dB")
    axes[1].legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    _style(axes[1])

    rot_names = ["pitch", "yaw", "roll"]
    trans_names = ["x", "y", "z"]
    for k, v in enumerate(versions):
        sim = sims[v]
        t = sim.motion_time
        m = sim.motion
        ax_r = axes[2 + 2 * k]
        ax_t = axes[3 + 2 * k]
        for j in range(3):
            ax_r.plot(t, np.degrees(m[:, j]), lw=1.0, color=SERIES[j], label=rot_names[j])
            ax_t.plot(t, m[:, 3 + j], lw=1.0, color=SERIES[j], label=trans_names[j])
        for ax in (ax_r, ax_t):
            ax.axhline(0, color=INK_MUTED, lw=0.4)
            for idx in feat.nucleus_idx:
                if idx < n:
                    ax.axvline(times[idx], color=SERIES[1], lw=0.3, alpha=0.3)
            _style(ax)
            ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=3)
        ax_r.set_ylabel(f"{labels[v]}\nrotation (deg)", fontsize=8)
        ax_t.set_ylabel(f"{labels[v]}\ntranslation (mm)", fontsize=8)
        ax_r.set_ylim(-30, 30)
        ax_t.set_ylim(-22, 22)

    axes[-1].set_xlabel("time (s)", color=INK)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def plot_summary(rows: list[LineResult], versions: list[str], out_path: Path) -> None:
    stems = list(dict.fromkeys(r.stem for r in rows))
    by = {(r.stem, r.version): r for r in rows}
    panels = [
        ("stillness_in_silence (higher is better)", "stillness", (0.0, 1.1)),
        ("voiced / unvoiced motion ratio (higher is better, capped at 5)", "vu_ratio", (0.0, 5.2)),
        ("onset_alignment (1 = chance, higher is better)", "onset_alignment", (0.0, 2.0)),
        ("rotation RMS over the line (deg)", "rms_rot_deg", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(stems))
    width = 0.8 / len(versions)
    for ax, (title, field, ylim) in zip(axes.flat, panels):
        for k, v in enumerate(versions):
            vals = []
            for s in stems:
                r = by.get((s, v))
                val = getattr(r, field) if r else np.nan
                if field == "vu_ratio":
                    val = min(val, 5.0)
                vals.append(val)
            ax.bar(x + (k - (len(versions) - 1) / 2) * width, vals, width * 0.92, color=SERIES[k], label=v, linewidth=0)
        ax.set_xticks(x)
        ax.set_xticklabels(stems, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        if ylim:
            ax.set_ylim(*ylim)
        if field == "onset_alignment":
            ax.axhline(1.0, color=INK_MUTED, lw=0.6, ls="--")
        _style(ax)
    axes[0, 0].legend(loc="lower left", fontsize=8, frameon=False, ncol=len(versions))
    emo = {r.stem: f"{r.emotion} {r.energy:.1f}" for r in rows if r.version == versions[-1]}
    sub = "   ".join(f"{s}: {emo.get(s, '')}" for s in stems)
    fig.suptitle("couple_fight lines, " + " / ".join(versions) + f" ({versions[-1]} with the beat's colouring)\n" + sub, fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def write_metrics_md(rows: list[LineResult], versions: list[str], out_path: Path) -> None:
    lines = [
        "# couple_fight lines: " + " / ".join(versions),
        "",
        "Metrics from `metrics.py` on the 50 ms hop grid. `stillness` is 1 - motion in silence / motion overall (1.0 = silences are still). "
        "`v/u` is the voiced / unvoiced motion ratio. `onset` is the onset alignment score (1.0 = chance). "
        "`peak rot` is the largest |pitch|, |yaw| or |roll| in degrees, `peak trans` the largest |x|, |y| or |z| in millimetres, "
        "`rot rms` the RMS of the three rotation axes over the whole line.",
        "",
        "| line | emotion | energy | version | stillness | v/u | onset | peak rot (deg) | peak trans (mm) | rot rms (deg) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        vu = "huge" if r.vu_ratio > 100 else f"{r.vu_ratio:.2f}"
        lines.append(
            f"| {r.stem} | {r.emotion} | {r.energy:.1f} | {r.version} | {r.stillness:.3f} | {vu} | {r.onset_alignment:.3f} "
            f"| {r.peak_rot_deg:.1f} | {r.peak_trans_mm:.1f} | {r.rms_rot_deg:.2f} |"
        )
    lines.append("")
    lines.append("Means over the lines:")
    lines.append("")
    lines.append("| version | stillness | onset | peak rot (deg) | rot rms (deg) |")
    lines.append("|---|---|---|---|---|")
    for v in versions:
        sub = [r for r in rows if r.version == v]
        if not sub:
            continue
        lines.append(
            f"| {v} | {np.mean([r.stillness for r in sub]):.3f} | {np.mean([r.onset_alignment for r in sub]):.3f} "
            f"| {np.mean([r.peak_rot_deg for r in sub]):.1f} | {np.mean([r.rms_rot_deg for r in sub]):.2f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument("--versions", default="v0,v5,v6")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "out" / "couple_fight")
    parser.add_argument("--exclude", default="scene_mix")
    args = parser.parse_args()

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}
    beats = load_scene_beats(args.scene)
    wavs = sorted(p for p in args.wav_dir.glob("*.wav") if p.stem not in exclude)
    if not wavs:
        parser.error(f"no wav in {args.wav_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[LineResult] = []
    for wav in wavs:
        pcm, sr = sim_mod.load_audio(str(wav))
        feat = feature_mod.extract(pcm, sr)
        beat = beats.get(wav.stem)
        sims: dict[str, sim_mod.SimResult] = {}
        labels: dict[str, str] = {}
        for v in versions:
            emotion, energy = colouring_for(v, beat, None, None)
            sim = sim_mod.run_tapper(v, pcm, sr, emotion=emotion, energy=energy)
            sims[v] = sim
            labels[v] = f"{v} ({emotion} {energy:.1f})" if sim_mod.supports_emotion(v) else v
            met = metric_mod.compute(sim.motion, feat.voiced, feat.rms_db, feat.onset_idx, sim.hop_ms)
            m = sim.motion
            rot = np.degrees(m[:, :3]) if len(m) else np.zeros((1, 3))
            trans = m[:, 3:] if len(m) else np.zeros((1, 3))
            rows.append(LineResult(
                stem=wav.stem, version=v, emotion=emotion, energy=energy,
                stillness=met.stillness_in_silence, vu_ratio=met.voiced_unvoiced_ratio,
                onset_alignment=met.onset_alignment,
                peak_rot_deg=float(np.abs(rot).max()), peak_trans_mm=float(np.abs(trans).max()),
                rms_rot_deg=float(np.sqrt(np.mean(rot ** 2))),
            ))
            print(f"{wav.stem:14s} {labels[v]:24s} still={met.stillness_in_silence:.3f} vu={met.voiced_unvoiced_ratio:7.2f} "
                  f"onset={met.onset_alignment:.3f} peak_rot={rows[-1].peak_rot_deg:5.1f}", flush=True)
        plot_line(wav.stem, pcm, sr, feat, sims, labels, args.out_dir / f"{wav.stem}.png")

    plot_summary(rows, versions, args.out_dir / "summary.png")
    write_metrics_md(rows, versions, args.out_dir / "metrics.md")
    print(f"wrote {args.out_dir / 'summary.png'} and {args.out_dir / 'metrics.md'}")


if __name__ == "__main__":
    main()
