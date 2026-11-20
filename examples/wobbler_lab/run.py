"""End-to-end driver: simulate, plot and score wobbler versions on speech clips.

Needs ``librosa``, ``scipy`` and ``matplotlib`` (offline only, the live
tappers use numpy alone). Run from the repo root on any speech recordings::

    python examples/wobbler_lab/run.py --wav speech.wav --version v0 v5 v6
    python examples/wobbler_lab/run.py --wav a.wav b.wav --version v6

One plot per clip and version is written to ``examples/wobbler_lab/out/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

from wobbler_lab import features as feature_mod  # noqa: E402
from wobbler_lab import metrics as metric_mod  # noqa: E402
from wobbler_lab import plot as plot_mod  # noqa: E402
from wobbler_lab import simulate as sim_mod  # noqa: E402


def run_one(wav: Path, version: str, out_dir: Path) -> dict[str, object]:
    """Simulate *version* on *wav*, write its plot, return its metrics row."""
    clip = wav.stem
    pcm, sr = sim_mod.load_audio(str(wav))
    sim = sim_mod.run_tapper(version, pcm, sr)
    feat = feature_mod.extract(pcm, sr)
    metrics = metric_mod.compute(
        sim.motion, feat.voiced, feat.rms_db, feat.onset_idx, sim.hop_ms,
    )
    plot_path = out_dir / f"{clip}_{version}.png"
    plot_mod.plot(sim, feat, version=version, clip=clip, out_path=plot_path)
    return {
        "clip": clip,
        "version": version,
        "stillness_in_silence": metrics.stillness_in_silence,
        "voiced_unvoiced_ratio": metrics.voiced_unvoiced_ratio,
        "onset_alignment": metrics.onset_alignment,
        "plot": str(plot_path),
    }


def main() -> None:
    """Parse the arguments, run every clip and version, print the metrics table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, nargs="+", required=True, help="speech recordings to analyse")
    parser.add_argument("--version", nargs="+", default=["v0", "v5", "v6"], choices=sorted(sim_mod.VERSIONS))
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for wav in args.wav:
        clip = wav.stem
        for version in args.version:
            print(f"-- running {clip} | {version} ...", flush=True)
            try:
                rows.append(run_one(wav, version, args.out))
            except Exception as e:
                print(f"   FAILED: {e}")
                rows.append({
                    "clip": clip, "version": version,
                    "stillness_in_silence": float("nan"),
                    "voiced_unvoiced_ratio": float("nan"),
                    "onset_alignment": float("nan"),
                    "plot": "(error)",
                })

    cols = ["clip", "version", "stillness_in_silence", "voiced_unvoiced_ratio", "onset_alignment", "plot"]
    print()
    header = "  ".join(f"{c:>22}" if c != "plot" else c for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:>22.3f}" if isinstance(v, float) else f"{v:>22}" if c != "plot" else str(v))
        print("  ".join(cells))


if __name__ == "__main__":
    main()
