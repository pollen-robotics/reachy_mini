"""Investigation tool: what does an antenna collision look like in the data?

HISTORY NOTE: this tool's stats explore the (v2) coupled-motion idea, which
Remi later rejected in favor of the geometric definition measured by hand
(see collision.py). Kept for the plots and as a record of the analysis.

Loads the HF recordings and prints/plots position and velocity signals.

Findings (2026-07-02, see README):
- diff = ant0 - ant1 is NOT a usable contact definition. The antennas are
  floppy friction-fit parts: their "rest" angles are wherever they were last
  left, so absolute thresholds break across robots and even across sessions
  (live robot rested at diff=+0.97 while the dataset robot rested at -0.35).
  Worse, the collision-definition recordings END with the antennas crossed
  and parked (diff ~ +5.5, not touching), a state indistinguishable by
  angles alone from a firm press.
- What IS distinctive is coupled motion: when the antennas touch, moving one
  moves the other. coupled speed m = min(|v0|, |v1|):
      rest / parked / single antenna moved :  m ~ 0
      slide/rub (touching, gentle)         :  m ~ 0.3 .. 1.2, sustained
      knock (audible collision)            :  m spikes 4 .. 9
  Each default*.json shows the bring-together clack + the 3 knocks as clean
  m-spikes; the 30 s of slide data contain zero spikes above 2.0.

Run:
    python examples/secret_handshake_lab/analyze_recordings.py [--plot]
"""

from __future__ import annotations

import argparse
import json
import os

REPO_ID = "RemiFabre/secret-handshake"

NAMES = [
    "default.json",
    "default2.json",
    "default3.json",
    "default4.json",
    "collision-definition.json",
    "collision-definition2.json",
]


def load_moves() -> dict[str, tuple[list[float], list[float], list[float]]]:
    from huggingface_hub import snapshot_download

    root = snapshot_download(repo_id=REPO_ID, repo_type="dataset")
    out = {}
    for name in NAMES:
        with open(os.path.join(root, "data", name)) as f:
            m = json.load(f)
        t0 = m["time"][0]
        ts = [x - t0 for x in m["time"]]
        a0 = [fr["antennas"][0] for fr in m["set_target_data"]]
        a1 = [fr["antennas"][1] for fr in m["set_target_data"]]
        out[name] = (ts, a0, a1)
    return out


def velocity(ts: list[float], xs: list[float], k: int = 2) -> list[float]:
    v = [0.0] * len(xs)
    for i in range(k, len(xs)):
        dt = ts[i] - ts[i - k]
        v[i] = (xs[i] - xs[i - k]) / dt if dt > 0 else 0.0
    return v


def coupled_speed(ts: list[float], a0: list[float], a1: list[float]) -> list[float]:
    v0 = velocity(ts, a0)
    v1 = velocity(ts, a1)
    return [min(abs(x), abs(y)) for x, y in zip(v0, v1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="save PNGs next to this script")
    args = parser.parse_args()

    moves = load_moves()

    print("old law (diff = ant0 - ant1) vs coupled-speed law (m = min(|v0|,|v1|)):\n")
    for name, (ts, a0, a1) in moves.items():
        d = [x - y for x, y in zip(a0, a1)]
        m = coupled_speed(ts, a0, a1)
        knocks = []
        last = -9.0
        for i in range(1, len(m)):
            if m[i] > 2.0 and m[i - 1] <= 2.0 and ts[i] - last > 0.25:
                knocks.append(round(ts[i], 2))
                last = ts[i]
        print(
            f"  {name:28s} diff range [{min(d):+.2f},{max(d):+.2f}]"
            f"  max_m={max(m):4.1f}  knocks(m>2.0)={len(knocks)} at {knocks}"
        )
    print(
        "\nExpected: 4-6 knocks on default*.json (bring-together clack + 3 taps,"
        "\nsometimes a bounce), ZERO on the collision-definition slides."
    )

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        here = os.path.dirname(os.path.abspath(__file__))
        for fname, series in (
            ("recordings_positions.png", "pos"),
            ("recordings_velocities.png", "vel"),
        ):
            fig, axes = plt.subplots(len(NAMES), 1, figsize=(14, 16))
            for ax, name in zip(axes, NAMES):
                ts, a0, a1 = moves[name]
                if series == "pos":
                    ax.plot(ts, a0, lw=1, label="ant0")
                    ax.plot(ts, a1, lw=1, label="ant1")
                    ax.plot(ts, [x - y for x, y in zip(a0, a1)], lw=1.4, color="k", label="diff")
                else:
                    ax.plot(ts, velocity(ts, a0), lw=0.9, label="v0")
                    ax.plot(ts, velocity(ts, a1), lw=0.9, label="v1")
                    ax.plot(ts, coupled_speed(ts, a0, a1), lw=1.4, color="k", label="m")
                    ax.axhline(2.0, color="r", ls="--", lw=0.7)
                ax.set_title(name)
                ax.legend(loc="upper right", fontsize=7)
                ax.grid(alpha=0.3)
            plt.tight_layout()
            path = os.path.join(here, fname)
            plt.savefig(path, dpi=90)
            print("saved", path)


if __name__ == "__main__":
    main()
