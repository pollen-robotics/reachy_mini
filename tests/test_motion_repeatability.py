"""Hardware-in-the-loop regression tests for dance repeatability."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
TOOLS_ROOT = REPO_ROOT / "tests" / "tools"
if TOOLS_ROOT.exists() and str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

try:
    from reachy_mini import ReachyMini
except ModuleNotFoundError as exc:  # pragma: no cover
    pytest.skip(
        f"reachy_mini is not importable: {exc}. Install the project dependencies (pip install -e .).",
        allow_module_level=True,
    )

from dance_measurement import (
    DanceMeasurementResult,
    DanceReference,
    MeasurementMode,
    measure_recorded_move,
)
from reachy_mini.motion.recorded_move import RecordedMoves

_DATASETS = {
    "dance": "pollen-robotics/reachy-mini-dances-library",
    "emotions": "pollen-robotics/reachy-mini-emotions-library",
}
_MOVES_PER_DATASET = 2
_REFERENCE_ROOT = Path(__file__).parent / "data" / "dance_references"
_ENV_MODE = "REACHY_MINI_REFERENCE_MODE"

pytestmark = pytest.mark.robot




def _resolve_mode(mini: ReachyMini, requested: str | None) -> MeasurementMode:
    if requested == "hardware":
        return MeasurementMode.HARDWARE
    if requested == "simulation":
        return MeasurementMode.SIMULATION
    sim_flag = _read_simulation_status(mini)
    if sim_flag is not None:
        return MeasurementMode.SIMULATION if sim_flag else MeasurementMode.HARDWARE
    inferred = getattr(mini, "use_sim", None)
    if inferred is None:
        raise RuntimeError(
            "Unable to infer mode automatically; set REACHY_MINI_REFERENCE_MODE to 'hardware' or 'simulation'."
        )
    return MeasurementMode.SIMULATION if inferred else MeasurementMode.HARDWARE


def _read_simulation_status(mini: ReachyMini) -> bool | None:
    try:
        status = mini.client.get_status(wait=True, timeout=2.0)
    except TimeoutError:
        return None
    sim_enabled = status.get("simulation_enabled") if isinstance(status, dict) else None
    if isinstance(sim_enabled, bool):
        return sim_enabled
    return None


def _load_reference(path: Path) -> DanceReference:
    if not path.exists():
        raise FileNotFoundError(
            f"Reference file {path} missing. Regenerate with python tests/tools/generate_motion_references.py."
        )
    try:
        with np.load(path, allow_pickle=False) as data:
            return DanceReference.from_npz(dict(data.items()))
    except ValueError as exc:
        raise RuntimeError(
            f"Reference file {path} uses a legacy format. Regenerate it with python tests/tools/generate_motion_references.py."
        ) from exc


def _assert_shapes(current: DanceMeasurementResult, reference: DanceReference) -> None:
    assert (
        current.goal_task_space.shape == reference.goal_task_space.shape
    ), "Requested task arrays changed size; regenerate references."
    assert (
        current.present_task_space.shape == reference.present_task_space.shape
    ), "Present task arrays changed size; regenerate references."
    assert (
        current.timestamps_s.shape == reference.timestamps_s.shape
    ), "Timing vector length mismatch; regenerate references."


def _verify_metrics(
    current: DanceMeasurementResult,
    reference: DanceReference,
    thresholds: dict[str, float],
) -> None:
    rms_limit = reference.metrics.rms_errors * thresholds["rms_multiplier"]
    worst_limit = reference.metrics.worst_errors * thresholds["worst_multiplier"]
    if not np.all(current.metrics.rms_errors <= rms_limit + 1e-12):
        viol = current.metrics.rms_errors - rms_limit
        viol = np.clip(viol, a_min=0.0, a_max=None)
        raise AssertionError(
            "RMS precision regression: "
            f"max excess {float(np.max(viol)):.3f} across {_describe_task_axes(viol)}"
        )
    if not np.all(current.metrics.worst_errors <= worst_limit + 1e-12):
        viol = current.metrics.worst_errors - worst_limit
        viol = np.clip(viol, a_min=0.0, a_max=None)
        raise AssertionError(
            "Worst-case precision regression: "
            f"max excess {float(np.max(viol)):.3f} across {_describe_task_axes(viol)}"
        )
    min_freq = thresholds.get("min_frequency_hz", 0.0)
    if current.metrics.average_update_frequency_hz + 1e-9 < min_freq:
        raise AssertionError(
            "Control-loop frequency too low: "
            f"{current.metrics.average_update_frequency_hz:.2f} Hz < {min_freq:.2f} Hz"
        )
    max_gap = thresholds.get("max_gap_s", 0.02)
    if current.metrics.max_update_gap_s - 1e-9 > max_gap:
        raise AssertionError(
            "Worst update gap too high: "
            f"{current.metrics.max_update_gap_s:.4f} s > {max_gap:.4f} s"
        )


def _describe_task_axes(values: np.ndarray) -> str:
    labels = (
        "x_mm",
        "y_mm",
        "z_mm",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "antenna_left_deg",
        "antenna_right_deg",
    )
    idx = int(np.argmax(np.abs(values)))
    return f"axis '{labels[idx]}'"


def test_dance_repeatability() -> None:
    requested_mode = os.getenv(_ENV_MODE)

    try:
        mini = ReachyMini()
    except (TimeoutError, ConnectionError) as exc:  # type: ignore[misc]
        pytest.skip(f"Unable to connect to Reachy Mini daemon: {exc}")

    with mini:
        mode = _resolve_mode(mini, requested_mode)
        mode_dir = _REFERENCE_ROOT / mode.value
        if not mode_dir.exists():
            pytest.skip(
                f"Reference directory {mode_dir} missing. Run python tests/tools/generate_motion_references.py first."
            )

        datasets = _select_dataset_moves(mode_dir, _MOVES_PER_DATASET)
        if not any(dataset_moves for dataset_moves in datasets.values()):
            pytest.skip(
                f"No reference files found in {mode_dir}. Generate references for at least one recorded move."
            )

        failures: list[str] = []

        for dataset_name, move_names in datasets.items():
            dataset_root = mode_dir / dataset_name
            if not move_names:
                failures.append(
                    f"Dataset '{dataset_name}' has no reference files in {dataset_root}."
                )
                continue
            try:
                recorded_library = RecordedMoves(_DATASETS[dataset_name])
            except Exception as exc:  # pragma: no cover
                failures.append(
                    f"Failed to load recorded moves dataset '{dataset_name}': {exc}"
                )
                continue

            for move_name in move_names:
                prefix = f"{dataset_name}/{move_name}"
                reference_path = dataset_root / f"{move_name}.npz"
                try:
                    reference = _load_reference(reference_path)
                except Exception as exc:  # pragma: no cover
                    failures.append(f"{prefix}: {exc}")
                    continue

                try:
                    recorded_move = recorded_library.get(move_name)
                except Exception as exc:
                    failures.append(f"{prefix}: unable to load move from dataset: {exc}")
                    continue

                try:
                    result = measure_recorded_move(
                        mini,
                        recorded_move,
                        move_name=move_name,
                        mode=mode,
                        config=reference.config,
                    )
                    _assert_shapes(result, reference)
                    _verify_metrics(result, reference, reference.thresholds)
                except AssertionError as exc:
                    failures.append(f"{prefix}: {exc}")
                except Exception as exc:
                    failures.append(f"{prefix}: unexpected error {exc}")

        if failures:
            details = "\n".join(failures)
            pytest.fail(
                "Motion repeatability regressions detected:\n" f"{details}",
                pytrace=False,
            )


def _select_dataset_moves(directory: Path, limit: int) -> dict[str, list[str]]:
    selections: dict[str, list[str]] = {}
    for dataset_name in _DATASETS:
        dataset_dir = directory / dataset_name
        if not dataset_dir.exists():
            selections[dataset_name] = []
            continue
        moves = sorted(p.stem for p in dataset_dir.glob("*.npz"))
        selections[dataset_name] = moves[:limit]
    return selections
