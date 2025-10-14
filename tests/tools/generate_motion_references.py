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
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves
from dance_measurement import (
    DanceMeasurementConfig,
    MeasurementMode,
    measure_recorded_move,
    reference_to_npz_payload,
)

_DATASETS = {
    "dance": "pollen-robotics/reachy-mini-dances-library",
    "emotions": "pollen-robotics/reachy-mini-emotions-library",
}
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
            "Record a subset of recorded moves and generate reference metrics "
            "for the motion repeatability regression test."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Directory where the .npz references will be saved.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(_DATASETS.keys()),
        choices=_DATASETS.keys(),
        help="Recorded move datasets to capture (defaults to both dance and emotions).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Maximum number of moves captured per dataset (defaults to 2).",
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

        target_dir = args.output_dir / mode.value
        target_dir.mkdir(parents=True, exist_ok=True)

        for dataset_name in args.datasets:
            dataset_dir = target_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            library = RecordedMoves(_DATASETS[dataset_name])
            move_names = sorted(library.list_moves())[: args.limit]

            for move_name in move_names:
                move = library.get(move_name)
                result = measure_recorded_move(
                    mini,
                    move,
                    move_name=move_name,
                    mode=mode,
                    config=config,
                )
                ref_thresholds = dict(thresholds)
                ref_thresholds["min_frequency_hz"] = max(
                    0.0,
                    result.metrics.average_update_frequency_hz - ref_thresholds["frequency_drop_hz"],
                )
                ref_thresholds["baseline_avg_frequency_hz"] = result.metrics.average_update_frequency_hz
                payload = reference_to_npz_payload(result, ref_thresholds)
                npz_path = dataset_dir / f"{move_name}.npz"
                np.savez_compressed(npz_path, **payload)
                print(
                    f"Saved reference for {move_name} in dataset {dataset_name} ({mode.value} mode) -> {npz_path} "
                    f"(samples={len(result.timestamps_s)})"
                )


if __name__ == "__main__":  # pragma: no cover
    main()
