"""End-to-end driver: simulate, plot, score wobbler versions on cached clips.

Run from the repo root::

    python -m examples.wobbler_lab.run --clip mlk --version v0 v1 v2 v3
    python -m examples.wobbler_lab.run --clip mlk kennedy degaulle --version v0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.wobbler_lab import features as feature_mod  # noqa: E402
from examples.wobbler_lab import metrics as metric_mod  # noqa: E402
from examples.wobbler_lab import plot as plot_mod  # noqa: E402
from examples.wobbler_lab import simulate as sim_mod  # noqa: E402

CLIPS = {
    "kennedy": ROOT / "examples" / ".tts_cache" / "kennedy.wav",
    "mlk": ROOT / "examples" / ".tts_cache" / "mlk.wav",
    "degaulle": ROOT / "examples" / ".tts_cache" / "degaulle.wav",
}


def run_one(clip: str, version: str, out_dir: Path) -> dict:
    pcm, sr = sim_mod.load_audio(str(CLIPS[clip]))
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
        "plot": str(plot_path.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", choices=list(CLIPS), nargs="+", default=["mlk"])
    parser.add_argument("--version", nargs="+", default=["v0", "v1", "v2", "v3"])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    args = parser.parse_args()

    rows = []
    for clip in args.clip:
        for version in args.version:
            print(f"-- running {clip} | {version} ...", flush=True)
            try:
                rows.append(run_one(clip, version, args.out))
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
