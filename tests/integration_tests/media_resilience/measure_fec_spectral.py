#!/usr/bin/env python3
"""Objectively quantify Opus in-band FEC recovery vs PLC, using a
phase-insensitive *spectral* metric on the deterministic chirp.

Why this works where waveform SNR failed:
  The reference is a frequency sweep (chirp). When a packet is lost:
    * PLC alone repeats the PREVIOUS frame -> wrong (stale) frequency.
    * In-band FEC reconstructs the frame from the next packet's redundancy
      -> correct frequency.
  So the *dominant frequency per 20ms frame* is a clean discriminator that
  does not care about exact sample phase (which concealment never matches).

Metric: for each config, compare the per-frame dominant frequency of the
lossy decode against its OWN clean decode, and report the mean absolute
frequency error (Hz). Lower = better recovery. FEC should beat PLC.

Reuses the loss-injection pipeline from measure_fec_quality.py.

Run with the venv that has gi + numpy:
  /tmp/venv/bin/python /tmp/measure_fec_spectral.py
"""

import argparse
import os
import sys

import numpy as np
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_fec_quality as q  # noqa: E402

RATE = 48000
FRAME = 960  # 20 ms


def to_np(arr):
    return np.frombuffer(bytes(arr), dtype=np.int16).astype(np.float64)


def dom_freq_track(samples):
    """Dominant frequency (Hz) per 20ms frame, DC removed, Hann-windowed."""
    n = (len(samples) // FRAME) * FRAME
    if n == 0:
        return np.zeros(0)
    frames = samples[:n].reshape(-1, FRAME)
    win = np.hanning(FRAME)
    spec = np.abs(np.fft.rfft(frames * win, axis=1))
    spec[:, 0] = 0.0  # kill DC
    bins = np.argmax(spec, axis=1)
    return bins * (RATE / FRAME)


def compare(ref_track, test_track):
    m = min(len(ref_track), len(test_track))
    if m == 0:
        return None, None, None
    r = ref_track[:m]
    t = test_track[:m]
    err = np.abs(r - t)
    mean_err = float(np.mean(err))
    # "affected" frames = where the lossy decode deviates noticeably
    affected = err > (RATE / FRAME)  # > 1 bin (50 Hz)
    n_aff = int(np.sum(affected))
    mean_err_aff = float(np.mean(err[affected])) if n_aff else 0.0
    return mean_err, n_aff, mean_err_aff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--losses", type=float, nargs="+", default=[5.0, 15.0, 30.0])
    args = ap.parse_args()

    Gst.init(None)
    print(f"[gen] chirp {args.seconds}s", flush=True)
    q.gen_reference(args.seconds)

    print("[ref] clean FEC off", flush=True)
    ref_off = dom_freq_track(to_np(q.Decode(False, 0.0, args.seed).run()))
    print("[ref] clean FEC on", flush=True)
    ref_on = dom_freq_track(to_np(q.Decode(True, 0.0, args.seed).run()))

    rows = []
    for loss in args.losses:
        seed = args.seed + int(loss)
        print(f"[test] loss {loss}%  FEC off", flush=True)
        toff = dom_freq_track(to_np(q.Decode(False, loss, seed).run()))
        print(f"[test] loss {loss}%  FEC on", flush=True)
        ton = dom_freq_track(to_np(q.Decode(True, loss, seed).run()))

        e_off, naff_off, eaff_off = compare(ref_off, toff)
        e_on, naff_on, eaff_on = compare(ref_on, ton)
        rows.append((loss, e_off, eaff_off, naff_off, e_on, eaff_on, naff_on))

    print()
    print("=" * 84)
    print("  FEC RECOVERY - per-frame dominant-frequency error vs clean decode (lower=better)")
    print("=" * 84)
    print(f"{'loss':>5} | {'PLC mean':>9} {'PLC@lost':>9} {'#lost':>6} | "
          f"{'FEC mean':>9} {'FEC@lost':>9} {'#lost':>6} | {'@lost gain':>11}")
    print("-" * 84)
    for loss, e_off, eaff_off, naff_off, e_on, eaff_on, naff_on in rows:
        gain = eaff_off - eaff_on
        print(f"{loss:>4.0f}% | "
              f"{e_off:>8.0f}Hz {eaff_off:>8.0f}Hz {naff_off:>6} | "
              f"{e_on:>8.0f}Hz {eaff_on:>8.0f}Hz {naff_on:>6} | "
              f"{gain:>+9.0f}Hz")
    print("=" * 84)
    print("PLC@lost / FEC@lost = mean frequency error on the frames that were actually")
    print("concealed. A large positive '@lost gain' = FEC reconstructed the correct")
    print("frequency where PLC repeated a stale one.")
    print()


if __name__ == "__main__":
    main()
