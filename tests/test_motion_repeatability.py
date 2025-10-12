"""Hardware-in-the-loop regression tests for dance repeatability."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

from reachy_mini import ReachyMini
from reachy_mini.testing.motion_capture import (
    DanceMeasurementResult,
    DanceReference,
    MeasurementMode,
    measure_single_dance,
)

_DEFAULT_MOVES = ("simple_nod", "side_to_side_sway")
_REFERENCE_ROOT = Path(__file__).parent / "data" / "dance_references"
_ENV_ENABLE = "REACHY_MINI_RUN_MOTION_TESTS"
_ENV_MODE = "REACHY_MINI_REFERENCE_MODE"


def _environment_enabled() -> bool:
    return os.getenv(_ENV_ENABLE, "0") == "1"


def _resolve_mode(mini: ReachyMini, requested: str | None) -> MeasurementMode:
    if requested == "hardware":
        return MeasurementMode.HARDWARE
    if requested == "simulation":
        return MeasurementMode.SIMULATION
    # Transition plan: when the daemon exposes simulation status via
    # ``mini.client.get_status()["simulation_enabled"]`` swap this attribute
    # access for the new API so we always rely on the backend's ground truth.
    inferred = getattr(mini, "use_sim", None)
    if inferred is None:
        raise RuntimeError(
            "ReachyMini instance does not expose 'use_sim'; specify mode via REACHY_MINI_REFERENCE_MODE."
        )
    return MeasurementMode.SIMULATION if inferred else MeasurementMode.HARDWARE


def _load_reference(path: Path) -> DanceReference:
    if not path.exists():
        raise FileNotFoundError(
            f"Reference file {path} missing. Regenerate with reachy-mini-generate-motion-reference."
        )
    with np.load(path, allow_pickle=False) as data:
        return DanceReference.from_npz(dict(data.items()))


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


def _verify_requested_targets(
    current: DanceMeasurementResult, reference: DanceReference, tolerance: float
) -> None:
    delta = np.abs(current.goal_task_space - reference.goal_task_space)
    max_delta = float(delta.max(initial=0.0))
    assert (
        max_delta <= tolerance
    ), f"Requested trajectories drifted (max Δ={max_delta:.3e} > tol={tolerance:.3e})."


def _verify_metrics(
    current: DanceMeasurementResult,
    reference: DanceReference,
    thresholds: dict[str, float],
) -> None:
    rms_limit = reference.metrics.rms_errors * thresholds["rms_multiplier"]
    worst_limit = reference.metrics.worst_errors * thresholds["worst_multiplier"]
    if not np.all(current.metrics.rms_errors <= rms_limit + 1e-12):
        viol = current.metrics.rms_errors - rms_limit
        raise AssertionError(
            "RMS precision regression: "
            f"max excess {float(np.max(viol)):.3f} across {_describe_task_axes(viol)}"
        )
    if not np.all(current.metrics.worst_errors <= worst_limit + 1e-12):
        viol = current.metrics.worst_errors - worst_limit
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


@pytest.mark.robot
@pytest.mark.skipif(  # type: ignore[arg-type]
    not _environment_enabled(),
    reason=f"Set {_ENV_ENABLE}=1 to enable motion repeatability tests.",
)
def test_dance_repeatability() -> None:
    requested_mode = os.getenv(_ENV_MODE)
    mode_dir = None
    with ReachyMini() as mini:
        mode = _resolve_mode(mini, requested_mode)
        mode_dir = _REFERENCE_ROOT / mode.value
        if not mode_dir.exists():
            pytest.skip(
                f"Reference directory {mode_dir} missing. Run reachy-mini-generate-motion-reference first."
            )
        moves = _select_moves(mode_dir, _DEFAULT_MOVES)
        if not moves:
            pytest.skip(
                f"No reference files found in {mode_dir}. Generate references for at least one move."
            )
        for move in moves:
            reference_path = mode_dir / f"{move}.npz"
            reference = _load_reference(reference_path)
            result = measure_single_dance(
                mini,
                move_name=move,
                mode=mode,
                config=reference.config,
            )
            _assert_shapes(result, reference)
            tolerance = reference.thresholds.get("goal_task_tolerance", 1e-6)
            _verify_requested_targets(result, reference, tolerance)
            _verify_metrics(result, reference, reference.thresholds)


def _select_moves(directory: Path, defaults: Iterable[str]) -> list[str]:
    available = {p.stem for p in directory.glob("*.npz")}
    selected = [m for m in defaults if m in available]
    if selected:
        return selected
    return sorted(available)
