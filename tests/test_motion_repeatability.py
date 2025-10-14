"""Hardware-in-the-loop regression tests for dance repeatability and precision.

The test deliberately drives the motion generator via ``set_target`` instead of
``ReachyMini.play_move`` – the latter currently exhibits timing bugs and does
not expose the commanded trajectory, which we need for precision checks.
"""

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
    _TASK_ORDER,
    measure_single_dance,
)

_DEFAULT_MOVES = ("simple_nod", "side_to_side_sway")
_REFERENCE_ROOT = Path(__file__).parent / "data" / "dance_references"
_ENV_MODE = "REACHY_MINI_REFERENCE_MODE"

# Tunable per-axis precision limits (first three entries in millimetres, the
# remaining in degrees). Adjust these values to tighten or relax the precision
# expectations without touching the verification logic below.
_PRECISION_RMS_LIMITS = {
    "x_mm": 1.0,
    "y_mm": 1.0,
    "z_mm": 1.0,
    "roll_deg": 1.0,
    "pitch_deg": 1.0,
    "yaw_deg": 1.0,
    "antenna_left_deg": 1.5,
    "antenna_right_deg": 1.5,
}
_PRECISION_WORST_LIMITS = {
    "x_mm": 2.0,
    "y_mm": 2.0,
    "z_mm": 2.0,
    "roll_deg": 2.0,
    "pitch_deg": 2.0,
    "yaw_deg": 2.0,
    "antenna_left_deg": 3.0,
    "antenna_right_deg": 3.0,
}
_DEFAULT_GOAL_DRIFT_LIMIT = 5e-3  # mm / deg, symmetric around stored goals

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
    """Compare measured metrics against the reference capture with symmetric tolerances.

    We treat repeatability as "stay close to the baseline" rather than absolute
    accuracy: each tolerance allows the metrics to drift a bit on either side of
    the recorded reference values.
    """

    rms_tolerance = _derive_tolerance(
        reference.metrics.rms_errors,
        multiplier=thresholds.get("rms_multiplier", 1.0),
        absolute=thresholds.get("rms_abs_tol", 0.0),
    )
    worst_tolerance = _derive_tolerance(
        reference.metrics.worst_errors,
        multiplier=thresholds.get("worst_multiplier", 1.0),
        absolute=thresholds.get("worst_abs_tol", 0.0),
    )

    rms_delta = np.abs(current.metrics.rms_errors - reference.metrics.rms_errors)
    if not np.all(rms_delta <= rms_tolerance + 1e-12):
        overrun = rms_delta - rms_tolerance
        overrun = np.clip(overrun, a_min=0.0, a_max=None)
        raise AssertionError(
            "RMS repeatability regression: "
            f"max excess {float(np.max(overrun)):.3f} across {_describe_task_axes(overrun)}"
        )
    worst_delta = np.abs(current.metrics.worst_errors - reference.metrics.worst_errors)
    if not np.all(worst_delta <= worst_tolerance + 1e-12):
        overrun = worst_delta - worst_tolerance
        overrun = np.clip(overrun, a_min=0.0, a_max=None)
        raise AssertionError(
            "Worst-case repeatability regression: "
            f"max excess {float(np.max(overrun)):.3f} across {_describe_task_axes(overrun)}"
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


def _verify_precision(
    tracking_rms: np.ndarray,
    tracking_worst: np.ndarray,
    goal_drift: np.ndarray,
    *,
    reference: DanceReference,
) -> None:
    """Ensure the live run meets precision requirements.

    Parameters are per-axis (aligned with ``_TASK_ORDER``) and expressed in mm or
    degrees as appropriate.
    """

    rms_limits = _axis_limits(_PRECISION_RMS_LIMITS)
    worst_limits = _axis_limits(_PRECISION_WORST_LIMITS)
    if not np.all(tracking_rms <= rms_limits + 1e-9):
        excess = np.clip(tracking_rms - rms_limits, a_min=0.0, a_max=None)
        raise AssertionError(
            "Precision RMS too high: "
            f"max excess {float(np.max(excess)):.3f} across {_describe_task_axes(excess)}"
        )
    if not np.all(tracking_worst <= worst_limits + 1e-9):
        excess = np.clip(tracking_worst - worst_limits, a_min=0.0, a_max=None)
        raise AssertionError(
            "Precision worst-case too high: "
            f"max excess {float(np.max(excess)):.3f} across {_describe_task_axes(excess)}"
        )

    goal_tolerance = float(reference.thresholds.get("goal_task_tolerance", _DEFAULT_GOAL_DRIFT_LIMIT))
    if not np.all(goal_drift <= goal_tolerance + 1e-9):
        excess = np.clip(goal_drift - goal_tolerance, a_min=0.0, a_max=None)
        raise AssertionError(
            "Commanded trajectory drifted: "
            f"max excess {float(np.max(excess)):.3f} across {_describe_task_axes(excess)}"
        )


def _derive_tolerance(
    reference_values: np.ndarray,
    *,
    multiplier: float,
    absolute: float,
) -> np.ndarray:
    """Compute symmetric tolerance (+/-) around the reference metric values.

    The multiplier encodes the relative margin (e.g. 1.20 == allow ±20% drift),
    while the absolute value protects axes whose reference is ~0 (still allow a
    tiny drift due to measurement noise).
    """

    multiplier = max(multiplier, 1.0)
    rel_margin = np.asarray(reference_values, dtype=float) * (multiplier - 1.0)
    abs_margin = np.full_like(rel_margin, fill_value=max(absolute, 0.0))
    return np.maximum(rel_margin, abs_margin)


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

        moves = _select_moves(mode_dir, _DEFAULT_MOVES)
        if not moves:
            pytest.skip(
                f"No reference files found in {mode_dir}. Generate references for at least one move."
            )

        failures: list[str] = []
        passes: list[str] = []

        for move_name in moves:
            reference_path = mode_dir / f"{move_name}.npz"
            try:
                reference = _load_reference(reference_path)
            except Exception as exc:  # pragma: no cover
                failures.append(f"{move_name}: {exc}")
                continue

            # Use the procedural sampling helper so we capture every command that
            # gets sent to the robot – this is critical for precision metrics.
            result = measure_single_dance(
                mini,
                move_name=move_name,
                mode=mode,
                config=reference.config,
            )
            _assert_shapes(result, reference)

            tracking_rms, tracking_worst, goal_drift = _compute_precision_metrics(result, reference)
            report = _format_metrics(
                move_name,
                result,
                reference,
                tracking_rms,
                tracking_worst,
                goal_drift,
            )

            try:
                _verify_precision(
                    tracking_rms,
                    tracking_worst,
                    goal_drift,
                    reference=reference,
                )
                _verify_metrics(result, reference, reference.thresholds)
            except AssertionError as exc:
                failures.append(f"{move_name}: {exc}\n{report}")
            except Exception as exc:
                failures.append(f"{move_name}: unexpected error {exc}\n{report}")
            else:
                passes.append(report)

        if failures:
            details = "\n".join(failures)
            summary = "\n\n".join(passes) if passes else "(no passing moves)"
            pytest.fail(
                "Motion repeatability regressions detected:\n"
                f"{details}\n\n"
                "Passing moves (latest metrics):\n"
                f"{summary}",
                pytrace=False,
            )
        for entry in passes:
            print(entry)


def _format_metrics(
    move_name: str,
    result: DanceMeasurementResult,
    reference: DanceReference,
    tracking_rms: np.ndarray,
    tracking_worst: np.ndarray,
    goal_drift: np.ndarray,
) -> str:
    """Human-readable report describing how the latest run compares to the baseline."""

    rms_repeat_tol = _derive_tolerance(
        reference.metrics.rms_errors,
        multiplier=reference.thresholds.get("rms_multiplier", 1.0),
        absolute=reference.thresholds.get("rms_abs_tol", 0.0),
    )
    worst_repeat_tol = _derive_tolerance(
        reference.metrics.worst_errors,
        multiplier=reference.thresholds.get("worst_multiplier", 1.0),
        absolute=reference.thresholds.get("worst_abs_tol", 0.0),
    )

    freq = float(result.metrics.average_update_frequency_hz)
    gap = float(result.metrics.max_update_gap_s)
    min_freq = reference.thresholds.get("min_frequency_hz", 0.0)
    max_gap = reference.thresholds.get("max_gap_s", 0.02)
    goal_tol = float(reference.thresholds.get("goal_task_tolerance", _DEFAULT_GOAL_DRIFT_LIMIT))

    precision_rms_limits = _axis_limits(_PRECISION_RMS_LIMITS)
    precision_worst_limits = _axis_limits(_PRECISION_WORST_LIMITS)

    def fmt_series(values: np.ndarray) -> str:
        return ", ".join(f"{axis}={value:.3f}" for axis, value in zip(_TASK_ORDER, values))

    return (
        f"{move_name} repeatability & precision summary:\n"
        "  Repeatability (actual vs. goal, compared to reference metrics):\n"
        f"    RMS reference:    {fmt_series(reference.metrics.rms_errors)}\n"
        f"    RMS tolerance ±:  {fmt_series(rms_repeat_tol)}\n"
        f"    RMS this run:     {fmt_series(result.metrics.rms_errors)}\n"
        f"    Worst reference:  {fmt_series(reference.metrics.worst_errors)}\n"
        f"    Worst tolerance ±:{fmt_series(worst_repeat_tol)}\n"
        f"    Worst this run:   {fmt_series(result.metrics.worst_errors)}\n"
        "  Precision (actual vs. goal within this run):\n"
        f"    RMS tracking:     {fmt_series(tracking_rms)} (limit {fmt_series(precision_rms_limits)})\n"
        f"    Worst tracking:   {fmt_series(tracking_worst)} (limit {fmt_series(precision_worst_limits)})\n"
        "  Command consistency (goal vs. stored reference goals):\n"
        f"    Max goal drift:   {fmt_series(goal_drift)} (limit ±{goal_tol:.3f})\n"
        "  Control loop timing:\n"
        f"    Average frequency: {freq:.2f} Hz (minimum {min_freq:.2f} Hz)\n"
        f"    Worst update gap: {gap:.4f} s (limit {max_gap:.4f} s)"
    )


def _select_moves(directory: Path, defaults: tuple[str, ...]) -> list[str]:
    available = {p.stem for p in directory.glob("*.npz")}
    chosen_defaults = [m for m in defaults if m in available]
    if chosen_defaults:
        return chosen_defaults
    return sorted(available)


def _compute_precision_metrics(
    result: DanceMeasurementResult,
    reference: DanceReference,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return tracking RMS/worst errors and command drift arrays (per axis).

    The output vectors are ordered according to ``_TASK_ORDER`` and therefore
    mix millimetres (XYZ) and degrees (rotations + antennas).
    """

    tracking_error = result.present_task_space - result.goal_task_space
    if tracking_error.size:
        rms = np.sqrt(np.mean(np.square(tracking_error), axis=0))
        worst = np.max(np.abs(tracking_error), axis=0)
    else:
        rms = np.zeros(len(_TASK_ORDER), dtype=float)
        worst = np.zeros(len(_TASK_ORDER), dtype=float)

    goal_delta = result.goal_task_space - reference.goal_task_space
    if goal_delta.size:
        drift = np.max(np.abs(goal_delta), axis=0)
    else:
        drift = np.zeros(len(_TASK_ORDER), dtype=float)

    return rms, worst, drift


def _axis_limits(mapping: dict[str, float]) -> np.ndarray:
    """Convert an axis->limit mapping into an array aligned with ``_TASK_ORDER``."""

    return np.asarray([mapping[axis] for axis in _TASK_ORDER], dtype=float)
