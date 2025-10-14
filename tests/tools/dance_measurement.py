"""Utilities to measure Reachy Mini dances for repeatability tests."""

from __future__ import annotations

import dataclasses
import enum
import json
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import numpy.typing as npt
from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini, utils
from reachy_mini.motion.recorded_move import RecordedMove

_NEUTRAL_POS = np.array([0.0, 0.0, 0.0], dtype=float)
_NEUTRAL_EUL = np.zeros(3, dtype=float)
_NEUTRAL_ANTENNAS = np.zeros(2, dtype=float)
_TASK_ORDER = (
    "x_mm",
    "y_mm",
    "z_mm",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "antenna_left_deg",
    "antenna_right_deg",
)


class MeasurementMode(str, enum.Enum):
    HARDWARE = "hardware"
    SIMULATION = "simulation"


@dataclass(slots=True)
class DanceMeasurementConfig:
    bpm: float = 120.0
    beats_per_move: float = 30.0
    sample_hz: float = 200.0
    initial_goto_duration: float = 1.0
    position_update_tol_m: float = 1e-5
    orientation_update_tol_rad: float = 1e-4
    antenna_update_tol_rad: float = 5e-4

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @staticmethod
    def from_json(data: str) -> "DanceMeasurementConfig":
        return DanceMeasurementConfig(**json.loads(data))


@dataclass(slots=True)
class DanceMetrics:
    rms_errors: npt.NDArray[np.float64]
    worst_errors: npt.NDArray[np.float64]
    average_update_frequency_hz: float
    max_update_gap_s: float

    def to_dict(self) -> Dict[str, float | list[float]]:
        return {
            "rms_errors": self.rms_errors.tolist(),
            "worst_errors": self.worst_errors.tolist(),
            "average_update_frequency_hz": float(self.average_update_frequency_hz),
            "max_update_gap_s": float(self.max_update_gap_s),
        }

    @staticmethod
    def from_dict(data: Dict[str, float | list[float]]) -> "DanceMetrics":
        return DanceMetrics(
            rms_errors=np.asarray(data["rms_errors"], dtype=np.float64),
            worst_errors=np.asarray(data["worst_errors"], dtype=np.float64),
            average_update_frequency_hz=float(data["average_update_frequency_hz"]),
            max_update_gap_s=float(data["max_update_gap_s"]),
        )


@dataclass(slots=True)
class DanceMeasurementResult:
    move_name: str
    config: DanceMeasurementConfig
    mode: MeasurementMode
    timestamps_s: npt.NDArray[np.float64]
    goal_pose_matrices: npt.NDArray[np.float64]
    present_pose_matrices: npt.NDArray[np.float64]
    goal_antennas_rad: npt.NDArray[np.float64]
    present_antennas_rad: npt.NDArray[np.float64]
    goal_task_space: npt.NDArray[np.float64]
    present_task_space: npt.NDArray[np.float64]
    metrics: DanceMetrics
    update_timestamps_s: npt.NDArray[np.float64]


@dataclass(slots=True)
class DanceReference:
    move_name: str
    mode: MeasurementMode
    config: DanceMeasurementConfig
    metrics: DanceMetrics
    thresholds: Dict[str, float]
    timestamps_s: npt.NDArray[np.float64]
    goal_pose_matrices: npt.NDArray[np.float64]
    present_pose_matrices: npt.NDArray[np.float64]
    goal_antennas_rad: npt.NDArray[np.float64]
    present_antennas_rad: npt.NDArray[np.float64]
    goal_task_space: npt.NDArray[np.float64]
    present_task_space: npt.NDArray[np.float64]
    update_timestamps_s: npt.NDArray[np.float64]

    @staticmethod
    def from_npz(data: Dict[str, np.ndarray]) -> "DanceReference":
        config = DanceMeasurementConfig.from_json(data["config_json"].item())
        thresholds = json.loads(data["thresholds_json"].item())
        metrics = DanceMetrics.from_dict(json.loads(data["metrics_json"].item()))
        return DanceReference(
            move_name=str(data["move_name"].item()),
            mode=MeasurementMode(data["mode"].item()),
            config=config,
            metrics=metrics,
            thresholds=thresholds,
            timestamps_s=np.asarray(data["timestamps_s"], dtype=np.float64),
            goal_pose_matrices=np.asarray(data["goal_pose_matrices"], dtype=np.float64),
            present_pose_matrices=np.asarray(data["present_pose_matrices"], dtype=np.float64),
            goal_antennas_rad=np.asarray(data["goal_antennas_rad"], dtype=np.float64),
            present_antennas_rad=np.asarray(data["present_antennas_rad"], dtype=np.float64),
            goal_task_space=np.asarray(data["goal_task_space"], dtype=np.float64),
            present_task_space=np.asarray(data["present_task_space"], dtype=np.float64),
            update_timestamps_s=np.asarray(data["update_timestamps_s"], dtype=np.float64),
        )


def measure_dances(
    mini: ReachyMini,
    move_names: Iterable[str],
    mode: MeasurementMode,
    config: DanceMeasurementConfig | None = None,
) -> Dict[str, DanceMeasurementResult]:
    cfg = config or DanceMeasurementConfig()
    return {
        move: measure_single_dance(mini, move, mode=mode, config=cfg)
        for move in move_names
    }


def measure_single_dance(
    mini: ReachyMini,
    move_name: str,
    mode: MeasurementMode,
    config: DanceMeasurementConfig | None = None,
) -> DanceMeasurementResult:
    cfg = config or DanceMeasurementConfig()
    if move_name not in AVAILABLE_MOVES:
        raise ValueError(f"Move '{move_name}' not available in dance library.")

    move_fn, base_params, _meta = AVAILABLE_MOVES[move_name]
    params: Dict[str, float] = dict(base_params)
    if "waveform" in params:
        params.setdefault("waveform", "sin")

    sample_period = 1.0 / cfg.sample_hz

    offsets = move_fn(0.0, **params)
    initial_pose = utils.create_head_pose(
        *(_NEUTRAL_POS + offsets.position_offset),
        *(_NEUTRAL_EUL + offsets.orientation_offset),
        degrees=False,
    )
    initial_antennas = _NEUTRAL_ANTENNAS + offsets.antennas_offset
    if cfg.initial_goto_duration > 0.0:
        mini.goto_target(head=initial_pose, antennas=initial_antennas, duration=cfg.initial_goto_duration)

    timestamps: list[float] = []
    goal_poses: list[npt.NDArray[np.float64]] = []
    present_poses: list[npt.NDArray[np.float64]] = []
    goal_antennas: list[npt.NDArray[np.float64]] = []
    present_antennas: list[npt.NDArray[np.float64]] = []
    goal_task: list[npt.NDArray[np.float64]] = []
    present_task: list[npt.NDArray[np.float64]] = []
    update_timestamps: list[float] = []

    prev_present_pose: npt.NDArray[np.float64] | None = None
    prev_present_ant: npt.NDArray[np.float64] | None = None

    start_perf = time.perf_counter()
    prev_tick = start_perf
    next_tick = start_perf
    beat_clock = 0.0

    while beat_clock < cfg.beats_per_move:
        next_tick += sample_period

        offsets = move_fn(beat_clock, **params)
        final_pos = _NEUTRAL_POS + offsets.position_offset
        final_eul = _NEUTRAL_EUL + offsets.orientation_offset
        final_ant = _NEUTRAL_ANTENNAS + offsets.antennas_offset

        target_pose = utils.create_head_pose(*final_pos, *final_eul, degrees=False)
        mini.set_target(target_pose, antennas=final_ant)
        present_pose = np.asarray(mini.get_current_head_pose(), dtype=float)
        present_ant = np.asarray(mini.get_present_antenna_joint_positions(), dtype=float)

        t_rel = time.perf_counter() - start_perf
        timestamps.append(t_rel)
        goal_poses.append(target_pose)
        present_poses.append(present_pose)
        goal_antennas.append(np.asarray(final_ant, dtype=float))
        present_antennas.append(present_ant)

        goal_task_vec = _pose_to_task_vector(target_pose, final_ant)
        present_task_vec = _pose_to_task_vector(present_pose, present_ant)
        goal_task.append(goal_task_vec)
        present_task.append(present_task_vec)

        if _is_new_update(
            present_pose,
            present_ant,
            prev_present_pose,
            prev_present_ant,
            position_tol=cfg.position_update_tol_m,
            orientation_tol=cfg.orientation_update_tol_rad,
            antenna_tol=cfg.antenna_update_tol_rad,
        ):
            update_timestamps.append(t_rel)
        prev_present_pose = present_pose
        prev_present_ant = present_ant

        remaining = next_tick - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        now = time.perf_counter()
        beat_clock += (now - prev_tick) * (cfg.bpm / 60.0)
        prev_tick = now

    timestamps_arr = np.asarray(timestamps, dtype=np.float64)
    goal_pose_arr = np.asarray(goal_poses, dtype=np.float64)
    present_pose_arr = np.asarray(present_poses, dtype=np.float64)
    goal_ant_arr = np.asarray(goal_antennas, dtype=np.float64)
    present_ant_arr = np.asarray(present_antennas, dtype=np.float64)
    goal_task_arr = np.asarray(goal_task, dtype=np.float64)
    present_task_arr = np.asarray(present_task, dtype=np.float64)
    update_arr = np.asarray(update_timestamps, dtype=np.float64)

    metrics = _compute_metrics(goal_task_arr, present_task_arr, timestamps_arr, update_arr)

    return DanceMeasurementResult(
        move_name=move_name,
        config=cfg,
        mode=mode,
        timestamps_s=timestamps_arr,
        goal_pose_matrices=goal_pose_arr,
        present_pose_matrices=present_pose_arr,
        goal_antennas_rad=goal_ant_arr,
        present_antennas_rad=present_ant_arr,
        goal_task_space=goal_task_arr,
        present_task_space=present_task_arr,
        metrics=metrics,
        update_timestamps_s=update_arr,
    )


def measure_recorded_move(
    mini: ReachyMini,
    move: RecordedMove,
    *,
    move_name: str,
    mode: MeasurementMode,
    config: DanceMeasurementConfig | None = None,
) -> DanceMeasurementResult:
    cfg = config or DanceMeasurementConfig()
    sample_period = 1.0 / cfg.sample_hz

    timestamps: list[float] = []
    goal_poses: list[npt.NDArray[np.float64]] = []
    present_poses: list[npt.NDArray[np.float64]] = []
    goal_antennas: list[npt.NDArray[np.float64]] = []
    present_antennas: list[npt.NDArray[np.float64]] = []
    goal_task: list[npt.NDArray[np.float64]] = []
    present_task: list[npt.NDArray[np.float64]] = []
    update_timestamps: list[float] = []

    prev_present_pose: npt.NDArray[np.float64] | None = None
    prev_present_ant: npt.NDArray[np.float64] | None = None

    start_pose, start_antennas, start_body_yaw = move.evaluate(0.0)
    if cfg.initial_goto_duration > 0.0:
        mini.goto_target(
            head=start_pose,
            antennas=start_antennas,
            duration=cfg.initial_goto_duration,
            body_yaw=start_body_yaw,
        )

    start_time = time.perf_counter()
    sample_index = 0

    while True:
        now = time.perf_counter()
        elapsed = now - start_time
        if elapsed >= move.duration:
            break

        eval_time = min(elapsed, move.duration - 1e-6)
        target_pose, target_antennas, target_body_yaw = move.evaluate(eval_time)

        mini.set_target(
            head=target_pose,
            antennas=target_antennas,
            body_yaw=target_body_yaw,
        )

        present_pose = np.asarray(mini.get_current_head_pose(), dtype=float)
        present_ant = np.asarray(mini.get_present_antenna_joint_positions(), dtype=float)

        timestamps.append(elapsed)
        goal_poses.append(np.asarray(target_pose, dtype=np.float64))
        present_poses.append(present_pose)
        goal_antennas.append(np.asarray(target_antennas, dtype=np.float64))
        present_antennas.append(present_ant)

        goal_task_vec = _pose_to_task_vector(target_pose, target_antennas)
        present_task_vec = _pose_to_task_vector(present_pose, present_ant)
        goal_task.append(goal_task_vec)
        present_task.append(present_task_vec)

        if _is_new_update(
            present_pose,
            present_ant,
            prev_present_pose,
            prev_present_ant,
            position_tol=cfg.position_update_tol_m,
            orientation_tol=cfg.orientation_update_tol_rad,
            antenna_tol=cfg.antenna_update_tol_rad,
        ):
            update_timestamps.append(elapsed)
        prev_present_pose = present_pose
        prev_present_ant = present_ant

        sample_index += 1
        target_next = start_time + sample_index * sample_period
        sleep_duration = target_next - time.perf_counter()
        if sleep_duration > 0.0:
            time.sleep(sleep_duration)

    timestamps_arr = np.asarray(timestamps, dtype=np.float64)
    goal_pose_arr = np.asarray(goal_poses, dtype=np.float64)
    present_pose_arr = np.asarray(present_poses, dtype=np.float64)
    goal_ant_arr = np.asarray(goal_antennas, dtype=np.float64)
    present_ant_arr = np.asarray(present_antennas, dtype=np.float64)
    goal_task_arr = np.asarray(goal_task, dtype=np.float64)
    present_task_arr = np.asarray(present_task, dtype=np.float64)
    update_arr = np.asarray(update_timestamps, dtype=np.float64)

    metrics = _compute_metrics(goal_task_arr, present_task_arr, timestamps_arr, update_arr)

    return DanceMeasurementResult(
        move_name=move_name,
        config=cfg,
        mode=mode,
        timestamps_s=timestamps_arr,
        goal_pose_matrices=goal_pose_arr,
        present_pose_matrices=present_pose_arr,
        goal_antennas_rad=goal_ant_arr,
        present_antennas_rad=present_ant_arr,
        goal_task_space=goal_task_arr,
        present_task_space=present_task_arr,
        metrics=metrics,
        update_timestamps_s=update_arr,
    )


def reference_to_npz_payload(
    result: DanceMeasurementResult, thresholds: Dict[str, float]
) -> Dict[str, npt.NDArray[np.float64] | np.ndarray]:
    payload: Dict[str, npt.NDArray[np.float64] | np.ndarray] = {
        "move_name": np.array(result.move_name, dtype=np.str_),
        "mode": np.array(result.mode.value, dtype=np.str_),
        "config_json": np.array(result.config.to_json(), dtype=np.str_),
        "thresholds_json": np.array(json.dumps(thresholds, sort_keys=True), dtype=np.str_),
        "metrics_json": np.array(json.dumps(result.metrics.to_dict(), sort_keys=True), dtype=np.str_),
        "timestamps_s": result.timestamps_s,
        "goal_pose_matrices": result.goal_pose_matrices,
        "present_pose_matrices": result.present_pose_matrices,
        "goal_antennas_rad": result.goal_antennas_rad,
        "present_antennas_rad": result.present_antennas_rad,
        "goal_task_space": result.goal_task_space,
        "present_task_space": result.present_task_space,
        "update_timestamps_s": result.update_timestamps_s,
    }
    return payload


def _pose_to_task_vector(
    pose: npt.NDArray[np.float64], antennas_rad: Sequence[float]
) -> npt.NDArray[np.float64]:
    pos_mm = pose[:3, 3] * 1000.0
    euler_deg = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
    ant_deg = np.rad2deg(antennas_rad)
    return np.concatenate([pos_mm, euler_deg, ant_deg]).astype(np.float64)


def _is_new_update(
    pose: npt.NDArray[np.float64],
    antennas: npt.NDArray[np.float64],
    prev_pose: npt.NDArray[np.float64] | None,
    prev_antennas: npt.NDArray[np.float64] | None,
    *,
    position_tol: float,
    orientation_tol: float,
    antenna_tol: float,
) -> bool:
    if prev_pose is None or prev_antennas is None:
        return True
    pos_delta = np.linalg.norm(pose[:3, 3] - prev_pose[:3, 3])
    rot_delta = R.from_matrix(prev_pose[:3, :3]).inv() * R.from_matrix(pose[:3, :3])
    ang_delta = np.linalg.norm(rot_delta.as_rotvec())
    antennas_delta = np.linalg.norm(antennas - prev_antennas)
    return (
        pos_delta > position_tol
        or ang_delta > orientation_tol
        or antennas_delta > antenna_tol
    )


def _compute_metrics(
    goal_task: npt.NDArray[np.float64],
    present_task: npt.NDArray[np.float64],
    timestamps: npt.NDArray[np.float64],
    update_timestamps: npt.NDArray[np.float64],
) -> DanceMetrics:
    errors = goal_task - present_task
    rms = (
        np.sqrt(np.mean(np.square(errors), axis=0))
        if errors.size
        else np.zeros(len(_TASK_ORDER), dtype=np.float64)
    )
    worst = (
        np.max(np.abs(errors), axis=0)
        if errors.size
        else np.zeros(len(_TASK_ORDER), dtype=np.float64)
    )

    if timestamps.size > 1:
        duration = float(timestamps[-1] - timestamps[0])
    else:
        duration = 0.0

    if update_timestamps.size > 1 and duration > 0.0:
        freq = float(update_timestamps.size / duration)
        gaps = np.diff(update_timestamps)
        max_gap = float(np.max(gaps)) if gaps.size else duration
    else:
        freq = 0.0
        max_gap = duration

    return DanceMetrics(
        rms_errors=rms,
        worst_errors=worst,
        average_update_frequency_hz=freq,
        max_update_gap_s=max_gap,
    )
