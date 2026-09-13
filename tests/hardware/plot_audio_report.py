"""Render report figures from the hardware audio test's artifacts.

The on-robot test writes ``audio.wav`` + ``curve.json`` when run with
``REACHY_TEST_ARTIFACTS=<dir>`` (matplotlib is not installed on the robot, so
plotting happens here, on a dev machine or CI runner)::

    uv run --with matplotlib python tests/hardware/plot_audio_report.py \
        artifacts_dir [more_dirs ...] -o report.png

One directory gives the standard report figure (response curve vs baseline with
tolerance band, plus a spectrogram of the recorded session). Several
directories overlay their curves — useful for tracking drift across CI runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+", type=Path, help="artifact directories")
    parser.add_argument("-o", "--out", type=Path, default=Path("audio_report.png"))
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.io import wavfile  # reads the IEEE-float WAVs wavenc produces

    runs = [(d, json.loads((d / "curve.json").read_text())) for d in args.dirs]
    first = runs[0][1]
    centers = first["band_centers_hz"]
    baseline = np.asarray(first["baseline_db"])
    tolerance = [t if t is not None else np.nan for t in first["tolerance_db"]]

    fig, (ax_curve, ax_spec) = plt.subplots(
        2, 1, figsize=(9, 8), height_ratios=[3, 2], constrained_layout=True
    )

    ax_curve.fill_between(
        centers,
        baseline - np.asarray(tolerance),
        baseline + np.asarray(tolerance),
        alpha=0.15,
        color="tab:green",
        label="tolerance",
    )
    ax_curve.plot(centers, baseline, "k--", lw=1.5, label="baseline")
    for directory, data in runs:
        verdict = "FAIL" if data["failures"] else "pass"
        ax_curve.plot(
            centers,
            data["curve_db"],
            marker="o",
            lw=1.8,
            label=f"{directory.name} ({verdict})",
        )
    ax_curve.set_xscale("log")
    ax_curve.set_xticks(centers, [str(c) for c in centers])
    ax_curve.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_curve.set_xlabel("band center (Hz)")
    ax_curve.set_ylabel("response (dB, median-normalized)")
    ax_curve.set_title("Speaker frequency response vs baseline")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend()

    rate, audio = wavfile.read(runs[-1][0] / "audio.wav")
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    ax_spec.specgram(mono, Fs=rate, NFFT=512, noverlap=384, cmap="magma")
    ax_spec.set_xlabel("time (s)")
    ax_spec.set_ylabel("frequency (Hz)")
    ax_spec.set_title(f"Recorded session ({runs[-1][0].name})")

    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
