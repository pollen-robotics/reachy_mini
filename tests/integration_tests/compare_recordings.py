#!/usr/bin/env python3
# compare_recordings.py
"""Compare two ReachyMini dance-run recordings by regenerating overlay plots.

Usage
-----
python compare_recordings.py measures/2025-08-07_11-36-00 measures/2025-08-27_15-30-30

What it does
------------
- Scans both input folders for per-move .npz files (created by your tracker).
- Keeps only the intersection of move names present in BOTH folders.
- For each common move:
  1) Recreates the 3-row error stack (translation [mm], angular [deg], combined [mm])
     and overlays the two runs.
  2) Recreates the 6-row XYZ/RPY stack. It overlays the "present" trajectories from
     both runs and shows the "goal" from run A as a thin reference line.
- Time axes are normalized per run (t - t[0]) and simply overlaid (no resampling).
- Saves results into: measures/compare_<runA>_vs_<runB>/

Notes
-----
- Expects the .npz schema produced by your data script:
  keys: t, trans_mm, ang_deg, magic_mm, goal_pos_m, present_pos_m,
        goal_rpy_deg, present_rpy_deg
- If a key is missing for any move in either run, that move is skipped with a warning.
- Matplotlib only, no seaborn. Plots use grids and legends; sizes chosen for readability.

Dependencies: numpy, matplotlib

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Set

import matplotlib.pyplot as plt
import numpy as np


# ------------------------- Logging ---------------------------------
def setup_logging() -> None:  # noqa: D103
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# ------------------------- I/O helpers ------------------------------
NPZ_REQUIRED_KEYS = {
    "t",
    "trans_mm",
    "ang_deg",
    "magic_mm",
    "goal_pos_m",
    "present_pos_m",
    "goal_rpy_deg",
    "present_rpy_deg",
}


def list_moves(folder: Path) -> Set[str]:
    """Return the set of move basenames (without extension) for all .npz files."""
    return {p.stem for p in folder.glob("*.npz") if p.is_file()}


def load_npz_safe(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load a .npz file and validate required keys. Returns dict or None."""
    try:
        with np.load(path) as data:
            keys = set(data.files)
            missing = NPZ_REQUIRED_KEYS - keys
            if missing:
                logging.warning("File %s missing keys: %s", path, sorted(missing))
                return None
            return {k: data[k] for k in NPZ_REQUIRED_KEYS}
    except Exception as e:
        logging.warning("Failed to load %s: %s", path, e)
        return None


# ------------------------- Plotting --------------------------------
def plot_errors_compare(
    A: Dict[str, np.ndarray],
    B: Dict[str, np.ndarray],
    move_name: str,
    out_png: Path,
) -> None:
    """Overlay error stacks (translation, angular, combined) for two runs."""
    tA = A["t"] - A["t"][0]
    tB = B["t"] - B["t"][0]

    fig, axes = plt.subplots(
        nrows=3, ncols=1, sharex=False, figsize=(12, 9), constrained_layout=True
    )

    ax = axes[0]
    ax.plot(tA, A["trans_mm"], linewidth=1.6, label="A trans_mm")
    ax.plot(tB, B["trans_mm"], linewidth=1.6, label="B trans_mm")
    ax.set_ylabel("Position error [mm]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(tA, A["ang_deg"], linewidth=1.6, label="A ang_deg")
    ax.plot(tB, B["ang_deg"], linewidth=1.6, label="B ang_deg")
    ax.set_ylabel("Angular error [deg]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[2]
    ax.plot(tA, A["magic_mm"], linewidth=1.6, label="A combined")
    ax.plot(tB, B["magic_mm"], linewidth=1.6, label="B combined")
    ax.set_ylabel("Combined [magic-mm]")
    ax.set_xlabel("Time [s] (each run normalized to start at 0)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.suptitle(
        f"Pose tracking errors vs time – compare A vs B – {move_name}", fontsize=14
    )
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_xyzrpy_compare(
    A: Dict[str, np.ndarray],
    B: Dict[str, np.ndarray],
    move_name: str,
    out_png: Path,
) -> None:
    """Overlay XYZ (mm) and RPY (deg) present trajectories for two runs.

    Also plot goal from run A as a thin reference line.
    """
    tA = A["t"] - A["t"][0]
    tB = B["t"] - B["t"][0]

    # Positions in mm
    goal_pos_A_mm = A["goal_pos_m"] * 1000.0
    present_A_mm = A["present_pos_m"] * 1000.0
    present_B_mm = B["present_pos_m"] * 1000.0

    # RPY in deg
    goal_rpy_A_deg = A["goal_rpy_deg"]
    present_A_rpy = A["present_rpy_deg"]
    present_B_rpy = B["present_rpy_deg"]

    labels = [
        ("X position [mm]", 0),
        ("Y position [mm]", 1),
        ("Z position [mm]", 2),
        ("Roll [deg]", 0),
        ("Pitch [deg]", 1),
        ("Yaw [deg]", 2),
    ]

    fig, axes = plt.subplots(
        nrows=6, ncols=1, sharex=False, figsize=(12, 14), constrained_layout=True
    )

    # XYZ: goal(A) thin, present(A), present(B)
    for ax, (ylabel, idx) in zip(axes[:3], labels[:3]):
        ax.plot(tA, goal_pos_A_mm[:, idx], linewidth=0.8, label=f"goal_A_{'xyz'[idx]}")
        ax.plot(
            tA, present_A_mm[:, idx], linewidth=1.6, label=f"present_A_{'xyz'[idx]}"
        )
        ax.plot(
            tB, present_B_mm[:, idx], linewidth=1.6, label=f"present_B_{'xyz'[idx]}"
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    # RPY: goal(A) thin, present(A), present(B)
    rpy_names = ["roll", "pitch", "yaw"]
    for ax, (ylabel, idx) in zip(axes[3:], labels[3:]):
        ax.plot(
            tA, goal_rpy_A_deg[:, idx], linewidth=0.8, label=f"goal_A_{rpy_names[idx]}"
        )
        ax.plot(
            tA,
            present_A_rpy[:, idx],
            linewidth=1.6,
            label=f"present_A_{rpy_names[idx]}",
        )
        ax.plot(
            tB,
            present_B_rpy[:, idx],
            linewidth=1.6,
            label=f"present_B_{rpy_names[idx]}",
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Time [s] (each run normalized to start at 0)")
    fig.suptitle(
        f"Head XYZ (mm) and RPY (deg) vs time – compare A vs B – {move_name}",
        fontsize=14,
    )
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ------------------------- Orchestration ----------------------------
def derive_output_root(dirA: Path, dirB: Path) -> Path:
    """Create output folder under the common 'measures' parent."""
    nameA = dirA.name
    nameB = dirB.name
    parentA = dirA.parent
    parentB = dirB.parent
    # Prefer parent of A if both look like 'measures'
    measures_parent = parentA if parentA.name == "measures" else parentA
    if parentB == parentA:
        measures_parent = parentA
    out = measures_parent / f"compare_{nameA}_vs_{nameB}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def process_move(move: str, dirA: Path, dirB: Path, out_dir: Path) -> None:  # noqa: D103
    pathA = dirA / f"{move}.npz"
    pathB = dirB / f"{move}.npz"

    dataA = load_npz_safe(pathA)
    dataB = load_npz_safe(pathB)
    if dataA is None or dataB is None:
        logging.warning("Skipping move '%s' due to load/keys issue.", move)
        return

    # Errors overlay
    out_err = out_dir / f"{move}_errors_compare.png"
    plot_errors_compare(dataA, dataB, move_name=move, out_png=out_err)

    # XYZ/RPY overlay
    out_xyzrpy = out_dir / f"{move}_xyzrpy_compare.png"
    plot_xyzrpy_compare(dataA, dataB, move_name=move, out_png=out_xyzrpy)

    logging.info("Saved %s and %s", out_err, out_xyzrpy)


def main() -> None:  # noqa: D103
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Regenerate and compare per-move plots from two recordings."
    )
    parser.add_argument(
        "dirA", type=Path, help="First run folder (e.g., measures/2025-08-07_11-36-00)"
    )
    parser.add_argument(
        "dirB", type=Path, help="Second run folder (e.g., measures/2025-08-27_15-30-30)"
    )
    args = parser.parse_args()

    dirA: Path = args.dirA
    dirB: Path = args.dirB

    if not dirA.is_dir() or not dirB.is_dir():
        logging.error("Both arguments must be existing directories.")
        return

    movesA = list_moves(dirA)
    movesB = list_moves(dirB)
    common = sorted(movesA & movesB)
    if not common:
        logging.error("No common moves found between %s and %s.", dirA, dirB)
        return

    out_dir = derive_output_root(dirA, dirB)
    logging.info("Common moves: %d. Output: %s", len(common), out_dir)

    for move in common:
        process_move(move, dirA, dirB, out_dir)

    logging.info("Done. Compared %d moves.", len(common))


if __name__ == "__main__":
    main()
