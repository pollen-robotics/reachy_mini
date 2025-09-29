#!/usr/bin/env python3
"""ReachyMini dance-run tracker: generate + measure at 200 Hz, per-move plots.

What it does
------------
- Iterates over all AVAILABLE_MOVES.
- For each move, runs it for a fixed number of beats at a fixed BPM.
- Single client loop: sends targets and measures current pose at 200 Hz.
- Errors via distance_between_poses: translation [mm], angular [deg], combined [magic-mm].
- Saves per-move data and two figures in a run folder:
    measures/YYYY-MM-DD_HH-MM-SS/<move>.npz
    measures/YYYY-MM-DD_HH-MM-SS/<move>_errors.png
    measures/YYYY-MM-DD_HH-MM-SS/<move>_xyzrpy_vs_time.png
- Logs a warning on every sample where the target pose equals the previous one.

Dependencies: numpy, matplotlib, scipy, reachy_mini
Style: ruff-compatible docstrings and type hints.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini, utils
from reachy_mini.utils.interpolation import distance_between_poses

# ---------------- Configuration (tweak as needed) ----------------
BPM: float = 120.0  # tempo for all moves
BEATS_PER_MOVE: float = 30.0  # duration per move
SAMPLE_HZ: float = 200.0  # control + measurement rate
NEUTRAL_POS = np.array([0.0, 0.0, 0.0])  # meters
NEUTRAL_EUL = np.zeros(3)  # radians
# -----------------------------------------------------------------


def setup_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )


def create_run_dir(base: Path = Path("measures")) -> Path:
    """Create and return a timestamped directory for this run."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = base / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_errors_stack(
    t_abs: np.ndarray,
    trans_mm: np.ndarray,
    ang_deg: np.ndarray,
    magic_mm: np.ndarray,
    title_suffix: str,
    out_png: Path,
    beat_period_s: float | None = None,
) -> None:
    """Create a 3-row vertical stack with shared X axis and save as PNG."""
    if t_abs.size == 0:
        logging.warning("No samples to plot for %s", title_suffix)
        return
    t = t_abs - t_abs[0]
    fig, axes = plt.subplots(
        nrows=3, ncols=1, sharex=True, figsize=(12, 9), constrained_layout=True
    )

    ax = axes[0]
    ax.plot(t, trans_mm, linewidth=1.6, label="|Δx|")
    ax.set_ylabel("Position error [mm]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(t, ang_deg, linewidth=1.6, label="|Δθ|")
    ax.set_ylabel("Angular error [deg]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[2]
    ax.plot(t, magic_mm, linewidth=1.6, label="mm + deg")
    ax.set_ylabel("Combined error [magic-mm]")
    ax.set_xlabel("Time [s]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    _draw_period_markers(axes, t, beat_period_s)

    fig.suptitle(f"Pose tracking errors vs time - {title_suffix}", fontsize=14)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_xyzrpy_stack(
    t_abs: np.ndarray,
    goal_pos_m: np.ndarray,
    present_pos_m: np.ndarray,
    goal_rpy_deg: np.ndarray,
    present_rpy_deg: np.ndarray,
    title_suffix: str,
    out_png: Path,
    beat_period_s: float | None = None,
) -> None:
    """Create a 6-row vertical stack (X/Y/Z in mm, Roll/Pitch/Yaw in deg), goal vs present."""
    if t_abs.size == 0:
        logging.warning("No samples to plot for %s", title_suffix)
        return
    t = t_abs - t_abs[0]

    # Convert positions to millimeters for plotting
    goal_pos_mm = goal_pos_m * 1000.0
    present_pos_mm = present_pos_m * 1000.0

    labels = [
        ("X position [mm]", 0),
        ("Y position [mm]", 1),
        ("Z position [mm]", 2),
        ("Roll [deg]", 0),
        ("Pitch [deg]", 1),
        ("Yaw [deg]", 2),
    ]

    fig, axes = plt.subplots(
        nrows=6, ncols=1, sharex=True, figsize=(12, 14), constrained_layout=True
    )

    # Positions (mm)
    for ax, (ylabel, idx) in zip(axes[:3], labels[:3]):
        ax.plot(t, goal_pos_mm[:, idx], linewidth=1.6, label=f"goal_{'xyz'[idx]}")
        ax.plot(t, present_pos_mm[:, idx], linewidth=1.6, label=f"present_{'xyz'[idx]}")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    # Orientations (deg)
    for ax, (ylabel, idx) in zip(axes[3:], labels[3:]):
        ax.plot(
            t,
            goal_rpy_deg[:, idx],
            linewidth=1.6,
            label=f"goal_{['roll', 'pitch', 'yaw'][idx]}",
        )
        ax.plot(
            t,
            present_rpy_deg[:, idx],
            linewidth=1.6,
            label=f"present_{['roll', 'pitch', 'yaw'][idx]}",
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Time [s]")
    _draw_period_markers(axes, t, beat_period_s)
    fig.suptitle(f"Head position and orientation vs time - {title_suffix}", fontsize=14)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _draw_period_markers(
    axes: np.ndarray, t: np.ndarray, beat_period_s: float | None
) -> None:
    if beat_period_s is None or beat_period_s <= 0.0 or t.size == 0:
        return
    duration = float(t[-1])
    if duration <= 0.0:
        return
    markers = np.arange(0.0, duration + 1e-9, beat_period_s)
    if markers.size == 0:
        markers = np.array([0.0])
    for ax in np.atleast_1d(axes):
        for marker in markers:
            ax.axvline(
                marker,
                color="tab:purple",
                linewidth=1.2,
                alpha=0.6,
                linestyle="--",
                zorder=3.0,
            )


def estimate_present_update_rate(
    t: np.ndarray, present_pos_m: np.ndarray, pos_tol_m: float = 1e-5
) -> float:
    """Estimate how often the present pose actually changes.

    We count samples where any XYZ component changes by more than pos_tol_m
    relative to the previous sample, then divide by total duration.

    Returns
    -------
    float
        Approximate update rate in Hz.

    """
    if t.size < 2:
        return 0.0
    diffs = np.abs(np.diff(present_pos_m, axis=0))
    changed = np.any(diffs > pos_tol_m, axis=1)
    n_changes = int(np.count_nonzero(changed))
    duration = float(t[-1] - t[0])
    return n_changes / duration if duration > 0 else 0.0


def save_npz(
    path: Path,
    data: Tuple[np.ndarray, ...],
    goal_pos_m: np.ndarray,
    present_pos_m: np.ndarray,
    goal_rpy_deg: np.ndarray,
    present_rpy_deg: np.ndarray,
) -> None:
    """Save measurements and extracted goal/present XYZ and RPY to .npz."""
    t, target, current, trans_mm, ang_deg, magic_mm = data
    np.savez_compressed(
        path,
        t=t,
        target=target,
        current=current,
        trans_mm=trans_mm,
        ang_deg=ang_deg,
        magic_mm=magic_mm,
        goal_pos_m=goal_pos_m,
        present_pos_m=present_pos_m,
        goal_rpy_deg=goal_rpy_deg,
        present_rpy_deg=present_rpy_deg,
    )


def run_one_move(
    mini: ReachyMini,
    move_name: str,
    move_def: Tuple,  # (move_fn, base_params, meta/desc)
    bpm: float,
    beats_total: float,
    sample_hz: float,
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate targets for a single move and measure tracking until beats_total.

    Returns
    -------
    data
        Tuple (t, target_poses, current_poses, trans_mm, ang_deg, magic_mm)
    goal_pos_m
        Array (N, 3) from target_pose[:3, 3].
    present_pos_m
        Array (N, 3) from current_pose[:3, 3].
    goal_rpy_deg
        Array (N, 3) from R.from_matrix(target_pose[:3,:3]).as_euler("xyz", degrees=True).
    present_rpy_deg
        Array (N, 3) from R.from_matrix(current_pose[:3,:3]).as_euler("xyz", degrees=True).

    """
    period = 1.0 / sample_hz
    move_fn, base_params, _ = move_def

    # Params: copy and set a default waveform if present
    params: Dict = dict(base_params)
    if "waveform" in params:
        params["waveform"] = params.get("waveform", "sin")

    # Buffers
    t_list: list[float] = []
    target_list: list[np.ndarray] = []
    current_list: list[np.ndarray] = []
    trans_list: list[float] = []
    ang_list: list[float] = []
    magic_list: list[float] = []
    goal_pos_list: list[np.ndarray] = []
    present_pos_list: list[np.ndarray] = []
    goal_rpy_list: list[np.ndarray] = []
    present_rpy_list: list[np.ndarray] = []

    # Initialize
    current_pose = np.asarray(mini.get_current_head_pose(), dtype=float)
    last_target = current_pose.copy()

    # Beat-time and scheduler
    t_beats = 0.0
    prev_tick = time.perf_counter()
    next_sched = prev_tick

    logging.info(
        "Move '%s' start (BPM=%.1f, duration=%.1f beats)", move_name, bpm, beats_total
    )
    while t_beats < beats_total:
        next_sched += period

        # Offsets at current beat time
        offsets = move_fn(t_beats, **params)
        final_pos = NEUTRAL_POS + offsets.position_offset
        final_eul = NEUTRAL_EUL + offsets.orientation_offset
        final_ant = offsets.antennas_offset

        # Send target and read back current
        target_pose = utils.create_head_pose(*final_pos, *final_eul, degrees=False)
        mini.set_target(target_pose, antennas=final_ant)
        current_pose = np.asarray(mini.get_current_head_pose(), dtype=float)

        # Warning on unchanged target
        if np.array_equal(target_pose, last_target):
            logging.warning("Target pose unchanged for move '%s'.", move_name)
        last_target = target_pose

        # Errors
        d_trans_m, d_ang_rad, d_magic_mm = distance_between_poses(
            target_pose, current_pose
        )

        # Extract XYZ and RPY from both goal and present directly from the matrices
        goal_pos = target_pose[:3, 3].astype(float)
        present_pos = current_pose[:3, 3].astype(float)
        goal_rpy = (
            R.from_matrix(target_pose[:3, :3])
            .as_euler("xyz", degrees=True)
            .astype(float)
        )
        present_rpy = (
            R.from_matrix(current_pose[:3, :3])
            .as_euler("xyz", degrees=True)
            .astype(float)
        )

        # Append
        t_list.append(time.time())
        target_list.append(target_pose)
        current_list.append(current_pose)
        trans_list.append(float(d_trans_m * 1000.0))
        ang_list.append(float(np.degrees(d_ang_rad)))
        magic_list.append(float(d_magic_mm))
        goal_pos_list.append(goal_pos)
        present_pos_list.append(present_pos)
        goal_rpy_list.append(goal_rpy)
        present_rpy_list.append(present_rpy)

        # Timing and beat advance
        remaining = next_sched - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        now = time.perf_counter()
        dt_real = now - prev_tick
        prev_tick = now
        t_beats += dt_real * (bpm / 60.0)

    # Convert to arrays
    t = np.asarray(t_list, dtype=float)
    target_arr = np.asarray(target_list, dtype=float)
    current_arr = np.asarray(current_list, dtype=float)
    trans_arr = np.asarray(trans_list, dtype=float)
    ang_arr = np.asarray(ang_list, dtype=float)
    magic_arr = np.asarray(magic_list, dtype=float)
    goal_pos_m = np.asarray(goal_pos_list, dtype=float)
    present_pos_m = np.asarray(present_pos_list, dtype=float)
    goal_rpy_deg = np.asarray(goal_rpy_list, dtype=float)
    present_rpy_deg = np.asarray(present_rpy_list, dtype=float)

    return (
        (t, target_arr, current_arr, trans_arr, ang_arr, magic_arr),
        goal_pos_m,
        present_pos_m,
        goal_rpy_deg,
        present_rpy_deg,
    )


def main() -> None:
    """Run all AVAILABLE_MOVES sequentially, saving per-move data and plots."""
    setup_logging()
    run_dir = create_run_dir(Path("measures"))

    with ReachyMini() as mini:
        mini.wake_up()
        try:
            for move_name, move_def in AVAILABLE_MOVES.items():
                data, goal_pos_m, present_pos_m, goal_rpy_deg, present_rpy_deg = (
                    run_one_move(
                        mini=mini,
                        move_name=move_name,
                        move_def=move_def,
                        bpm=BPM,
                        beats_total=BEATS_PER_MOVE,
                        sample_hz=SAMPLE_HZ,
                    )
                )

                rate_hz = estimate_present_update_rate(
                    data[0], present_pos_m, pos_tol_m=1e-5
                )
                logging.info(
                    "Estimated present pose update rate for '%s': %.1f Hz",
                    move_name,
                    rate_hz,
                )

                # Save data and plots for this move
                npz_path = run_dir / f"{move_name}.npz"
                png_errors = run_dir / f"{move_name}_errors.png"
                png_xyzrpy = run_dir / f"{move_name}_xyzrpy_vs_time.png"

                save_npz(
                    npz_path,
                    data,
                    goal_pos_m,
                    present_pos_m,
                    goal_rpy_deg,
                    present_rpy_deg,
                )
                t, _target, _current, trans_mm, ang_deg, magic_mm = data
                beat_period_s = 60.0 / BPM if BPM > 0 else None

                plot_errors_stack(
                    t_abs=t,
                    trans_mm=trans_mm,
                    ang_deg=ang_deg,
                    magic_mm=magic_mm,
                    title_suffix=move_name,
                    out_png=png_errors,
                    beat_period_s=beat_period_s,
                )
                plot_xyzrpy_stack(
                    t_abs=t,
                    goal_pos_m=goal_pos_m,
                    present_pos_m=present_pos_m,
                    goal_rpy_deg=goal_rpy_deg,
                    present_rpy_deg=present_rpy_deg,
                    title_suffix=move_name,
                    out_png=png_xyzrpy,
                    beat_period_s=beat_period_s,
                )

                logging.info("Saved %s, %s and %s", npz_path, png_errors, png_xyzrpy)
        except KeyboardInterrupt:
            logging.info(
                "Interrupted by user. Finishing current move and saving what is available."
            )
        finally:
            mini.goto_sleep()
            logging.info("Run folder: %s", run_dir)


if __name__ == "__main__":
    main()
