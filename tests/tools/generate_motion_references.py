#!/usr/bin/env python3
"""Generate motion repeatability references for Reachy Mini dances."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reachy_mini import ReachyMini
from reachy_mini.testing.motion_capture import (
    DanceMeasurementConfig,
    MeasurementMode,
    measure_dances,
)

_DEFAULT_MOVES = ("simple_nod", "side_to_side_sway")
_DEFAULT_OUTPUT = REPO_ROOT / "tests" / "data" / "dance_references"
_DEFAULT_THRESHOLDS: Dict[MeasurementMode, Dict[str, float]] = {
    MeasurementMode.HARDWARE: {
        "rms_multiplier": 1.20,
        "worst_multiplier": 1.30,
        "frequency_drop_hz": 10.0,
        "max_gap_s": 0.02,
        "goal_task_tolerance": 1e-5,
    },
    MeasurementMode.SIMULATION: {
        "rms_multiplier": 1.10,
        "worst_multiplier": 1.15,
        "frequency_drop_hz": 4.0,
        "max_gap_s": 0.02,
        "goal_task_tolerance": 1e-6,
    },
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record a subset of dances and generate reference metrics for the "
            "motion repeatability regression test."
        )
    )
    parser.add_argument(
        "--moves",
        nargs="*",
        default=_DEFAULT_MOVES,
        help="Dance names to record (defaults to simple_nod and side_to_side_sway).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Directory where the .npz references will be saved.",
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=120.0,
        help="Tempo used when generating targets (beats per minute).",
    )
    parser.add_argument(
        "--beats",
        type=float,
        default=30.0,
        help="Number of beats captured for each move.",
    )
    parser.add_argument(
        "--sample-hz",
        type=float,
        default=200.0,
        help="Sampling rate (Hz) for command + measurement loop.",
    )
    parser.add_argument(
        "--initial-goto",
        type=float,
        default=1.0,
        help="Duration of the initial goto to align on the move's starting pose.",
    )
    return parser.parse_args(argv)


def _determine_mode(reachy: ReachyMini) -> MeasurementMode:
    try:
        status = reachy.client.get_status(wait=True, timeout=2.0)
    except TimeoutError:
        status = None
    sim_enabled = None
    if isinstance(status, dict):
        sim_enabled = status.get("simulation_enabled")
    if isinstance(sim_enabled, bool):
        return MeasurementMode.SIMULATION if sim_enabled else MeasurementMode.HARDWARE
    inferred = getattr(reachy, "use_sim", None)
    if inferred is not None:
        return MeasurementMode.SIMULATION if inferred else MeasurementMode.HARDWARE
    raise RuntimeError("Unable to determine if the daemon runs in simulation or on hardware.")


def _build_thresholds(mode: MeasurementMode) -> Dict[str, float]:
    return dict(_DEFAULT_THRESHOLDS[mode])


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config = DanceMeasurementConfig(
        bpm=args.bpm,
        beats_per_move=args.beats,
        sample_hz=args.sample_hz,
        initial_goto_duration=args.initial_goto,
    )

    with ReachyMini() as mini:
        mode = _determine_mode(mini)
        thresholds = _build_thresholds(mode)
        results = measure_dances(mini, args.moves, mode=mode, config=config)

    target_dir = args.output_dir / mode.value
    target_dir.mkdir(parents=True, exist_ok=True)

    for move_name, result in results.items():
        ref_thresholds = dict(thresholds)
        ref_thresholds["min_frequency_hz"] = max(
            0.0,
            result.metrics.average_update_frequency_hz - ref_thresholds["frequency_drop_hz"],
        )
        ref_thresholds["baseline_avg_frequency_hz"] = result.metrics.average_update_frequency_hz
        reference = result.to_reference(ref_thresholds)
        npz_path = target_dir / f"{move_name}.npz"
        np.savez_compressed(npz_path, **reference.to_npz_payload())
        print(
            f"Saved reference for {move_name} in {mode.value} mode -> {npz_path} "
            f"(samples={len(result.timestamps_s)})"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
