#!/usr/bin/env python3
"""Derive Reachy Mini speaker-EQ gains from hifiscan measurements.

The plastic head shell colors the speaker output ("boxy": low-mid resonances +
comb peaks). A GStreamer ``equalizer-10bands`` on the daemon's speaker branch
corrects it; this offline tool computes the 10 per-band gains (dB) to put under
``speaker_eq_gains`` in the daemon config (``daemon_config.json``). It is not
used at runtime and users are unlikely to need it. See ``README.md``.

Workflow:

1. ``uv pip install hifiscan`` and connect an *external* mic at a fixed
   listening position (not the robot's own mic; its XVF3800 DSP colors the
   measurement). Because we take a *differential*, the mic/amp/room coloration
   cancels, so a calibrated mic is not required, only that nothing moves.
2. Run hifiscan and sweep the robot speaker twice, same volume/position,
   averaging several sweeps each (a single sweep is noisy):
     - shell **on**  (assembled robot) -> hifiscan "save" -> ``corr_on.pkl``
     - shell **off** (speaker in its bare case)          -> ``corr_off.pkl``
3. Derive the gains::

     uv run python src/reachy_mini/tools/speaker_eq_calibration/calibrate.py \
         gains corr_on.pkl corr_off.pkl

   Put the printed list under ``speaker_eq_gains`` in the daemon config
   (``~/.config/reachy_mini/daemon_config.json``), or bake it into
   ``reachy_mini.media.audio_utils.DEFAULT_SPEAKER_EQ_GAINS``.
4. Optionally render the calibration figure (needs matplotlib)::

     uv run --with matplotlib python \
         src/reachy_mini/tools/speaker_eq_calibration/calibrate.py plot \
         corr_on.pkl corr_off.pkl corr_on_16k.pkl corr_off_16k.pkl \
         -o docs/assets/speaker_eq_calibration.png

Caveats: a 10-band graphic EQ has fixed octave bands, so a narrow resonance may
fall between centers; it corrects the magnitude response only (not time-domain
ringing or mechanical buzz); and on the 16 kHz voice path only bands <= 8 kHz
carry signal (top bands are zeroed via ``--zero-above``).
"""

import argparse

import numpy as np
import numpy.typing as npt

# equalizer-10bands ISO octave band centers (Hz) and per-band gain range (dB).
BAND_CENTERS = (
    29.0,
    59.0,
    119.0,
    237.0,
    474.0,
    947.0,
    1889.0,
    3770.0,
    7523.0,
    15011.0,
)
GAIN_MIN, GAIN_MAX = -24.0, 12.0

_SMOOTH = 15.0  # hifiscan spectral smoothing before octave-band averaging


def bin_to_bands(
    freqs: npt.NDArray[np.float64], mag_db: npt.NDArray[np.float64]
) -> list[float]:
    """Average a magnitude-response curve (dB vs Hz) into the 10 EQ bands.

    Each band takes the mean dB over a +/- half-octave window around its center.
    """
    out = []
    for center in BAND_CENTERS:
        lo, hi = center / 2**0.5, center * 2**0.5
        mask = (freqs >= lo) & (freqs < hi)
        out.append(float(np.mean(mag_db[mask])) if mask.any() else 0.0)
    return out


def band_gains_from_responses(
    freqs: npt.NDArray[np.float64],
    resp_on_db: npt.NDArray[np.float64],
    resp_off_db: npt.NDArray[np.float64],
    zero_above: float | None = None,
    max_boost: float = GAIN_MAX,
) -> list[float]:
    """Compute the 10 EQ band gains (dB) that cancel the shell's coloration.

    ``resp_on_db`` / ``resp_off_db`` are the measured magnitude responses (dB) of
    the speaker with the head shell on and off. What the shell adds is::

        coloration(f) = resp_on(f) - resp_off(f)

    and the correcting gain is its negative. We recenter by the median of the
    trusted bands (the two measurements differ by an arbitrary overall level we
    must not apply as broadband gain), clamp to the element's range, and zero
    bands above ``zero_above`` (untrusted treble and/or beyond the voice-path
    Nyquist). ``max_boost`` caps positive gains (noisy differentials can ask for
    unrealistic boosts that only add hiss and eat headroom).
    """
    gains = bin_to_bands(freqs, resp_off_db - resp_on_db)  # = -coloration

    trusted = [
        g
        for center, g in zip(BAND_CENTERS, gains)
        if zero_above is None or center <= zero_above
    ]
    if trusted:
        offset = float(np.median(trusted))
        gains = [g - offset for g in gains]

    ceiling = min(max_boost, GAIN_MAX)
    clamped = [min(max(g, GAIN_MIN), ceiling) for g in gains]
    if zero_above is not None:
        clamped = [
            g if center <= zero_above else 0.0
            for center, g in zip(BAND_CENTERS, clamped)
        ]
    return clamped


def _load_response(
    path: str, smoothing: float = _SMOOTH
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return (freqs, magnitude_dB) from a hifiscan "save" (pickled Analyzer)."""
    import pickle

    with open(path, "rb") as fh:
        analyzer = pickle.load(fh)
    spectrum = analyzer.spectrum(smoothing)
    return spectrum.x, spectrum.y


def _coloration(
    on: str, off: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return (freqs, coloration_dB) with the overall level removed.

    Centered on the band-median (<=8 kHz) so it mirrors the net-zero EQ.
    """
    freqs, resp_on = _load_response(on)
    freqs_off, resp_off = _load_response(off)
    col = resp_on - np.interp(freqs, freqs_off, resp_off)
    bands = bin_to_bands(freqs, col)
    med = float(np.median([g for c, g in zip(BAND_CENTERS, bands) if c <= 8000]))
    return freqs, col - med


def _cmd_gains(
    on: str, off: str, zero_above: float, max_boost: float, smoothing: float
) -> None:
    """Print the 10 EQ band gains and a daemon_config.json snippet."""
    freqs_on, resp_on = _load_response(on, smoothing)
    freqs_off, resp_off = _load_response(off, smoothing)
    if not np.array_equal(freqs_on, freqs_off):  # same chirp/rate -> same axis
        resp_off = np.interp(freqs_on, freqs_off, resp_off)

    gains = band_gains_from_responses(
        freqs_on, resp_on, resp_off, zero_above=zero_above, max_boost=max_boost
    )
    print("Band gains (dB):")
    for center, gain in zip(BAND_CENTERS, gains):
        print(f"  {center:8.0f} Hz : {gain:+6.2f}")
    snippet = ", ".join(f"{g:.2f}" for g in gains)
    print(f'\ndaemon_config.json:  "speaker_eq_gains": [{snippet}]')


def _cmd_plot(on44: str, off44: str, on16: str, off16: str, out: str) -> None:
    """Render the two-panel calibration figure (measured response + correction)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref, boxy, fix, col_c = "#5b6b7a", "#d1495b", "#2a9d8f", "#e9c46a"

    # Panel 1: raw responses (44.1 kHz), level-matched over 100 Hz-8 kHz.
    freqs, resp_on = _load_response(on44)
    freqs_off, resp_off = _load_response(off44)
    resp_off = np.interp(freqs, freqs_off, resp_off)
    band = (freqs >= 100) & (freqs <= 8000)
    resp_on = resp_on - resp_on[band].mean()
    resp_off = resp_off - resp_off[band].mean()

    # Panel 2: coloration (level-removed) + applied EQ (16 kHz voice set).
    fc44, col44 = _coloration(on44, off44)
    fc16, col16 = _coloration(on16, off16)
    f16, y16on = _load_response(on16)
    fo16, y16off = _load_response(off16)
    eq = band_gains_from_responses(
        f16, y16on, np.interp(f16, fo16, y16off), zero_above=8000.0
    )
    edges, steps = [], []
    for center, gain in zip(BAND_CENTERS, eq):
        edges += [center / 2**0.5, center * 2**0.5]
        steps += [gain, gain]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.semilogx(freqs, resp_off, color=ref, lw=1.8, label="Shell off (bare case)")
    ax1.semilogx(
        freqs, resp_on, color=boxy, lw=1.8, label="Shell on (assembled — boxy)"
    )
    ax1.set_title("Reachy Mini speaker: measured response (44.1 kHz, level-matched)")
    ax1.set_ylabel("Relative level (dB)")
    ax1.legend(loc="lower center", ncol=2, frameon=False)
    ax1.grid(True, which="both", alpha=0.25)

    ax2.axhline(0, color="#999", lw=0.8)
    ax2.fill_between(
        fc16, col16, color=col_c, alpha=0.45, label="Shell coloration, 16 kHz"
    )
    ax2.semilogx(
        fc44, col44, color="#8a6d3b", lw=1.0, ls=":", label="Shell coloration, 44.1 kHz"
    )
    ax2.step(edges, steps, where="post", color=fix, lw=2.2, label="Applied 10-band EQ")
    for center in BAND_CENTERS:
        ax2.axvline(center, color="#ccc", lw=0.5, alpha=0.5)
    ax2.set_title("Shell coloration vs. applied correction (EQ ≈ −coloration)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Gain (dB)")
    ax2.set_xlim(20, 20000)
    ax2.legend(loc="lower center", ncol=2, frameon=False)
    ax2.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("saved", out)


def main() -> None:
    """CLI: derive EQ gains from, or plot, two hifiscan measurements."""
    parser = argparse.ArgumentParser(
        description="Reachy Mini speaker EQ calibration (offline reference tool)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gains", help="print the 10 EQ band gains")
    g.add_argument("on", help="hifiscan measurement, shell ON")
    g.add_argument("off", help="hifiscan measurement, shell OFF")
    g.add_argument("--zero-above", type=float, default=8000.0)
    g.add_argument("--max-boost", type=float, default=12.0)
    g.add_argument("--smoothing", type=float, default=_SMOOTH)

    p = sub.add_parser("plot", help="render the calibration figure (needs matplotlib)")
    p.add_argument("on44")
    p.add_argument("off44")
    p.add_argument("on16")
    p.add_argument("off16")
    p.add_argument("-o", "--out", default="docs/assets/speaker_eq_calibration.png")

    args = parser.parse_args()
    if args.cmd == "gains":
        _cmd_gains(args.on, args.off, args.zero_above, args.max_boost, args.smoothing)
    else:
        _cmd_plot(args.on44, args.off44, args.on16, args.off16, args.out)


if __name__ == "__main__":
    main()
