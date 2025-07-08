from dataclasses import dataclass

import numpy as np


@dataclass
class MoveOffsets:
    position_offset: np.ndarray
    orientation_offset: np.ndarray
    antennas_offset: np.ndarray


def oscillation_motion(
    t_beats: float,
    amplitude: float,
    cycles_per_beat: float = 1.0,
    phase_offset: float = 0.0,
    waveform: str = "sin",
) -> float:
    phase = 2 * np.pi * (cycles_per_beat * t_beats + phase_offset)
    if waveform == "sin":
        return amplitude * np.sin(phase)
    elif waveform == "cos":
        return amplitude * np.cos(phase)
    elif waveform == "square":
        return amplitude * np.sign(np.sin(phase))
    elif waveform == "triangle":
        return amplitude * (2 / np.pi) * np.arcsin(np.sin(phase))
    elif waveform == "sawtooth":
        return amplitude * (2 * ((phase / (2 * np.pi)) % 1) - 1)
    else:
        raise ValueError(f"Unsupported waveform type: {waveform}")


def transient_motion(
    t_beats: float,
    amplitude: float,
    duration_beats: float = 1.0,
    delay_beats: float = 0.0,
    repeat_every: float = 0.0,
):
    if repeat_every <= 0.0:
        repeat_every = duration_beats + delay_beats
    start_time = (
        np.floor((t_beats - delay_beats) / repeat_every) * repeat_every + delay_beats
    )
    if start_time <= t_beats < start_time + duration_beats:
        t_norm = (t_beats - start_time) / duration_beats
        eased_t = t_norm * t_norm * (3.0 - 2.0 * t_norm)
        return amplitude * eased_t
    return 0.0


def combine_offsets(*offsets: MoveOffsets) -> MoveOffsets:
    if not offsets:
        return MoveOffsets(np.zeros(3), np.zeros(3), np.zeros(2))
    pos = sum((o.position_offset for o in offsets), np.zeros(3))
    ori = sum((o.orientation_offset for o in offsets), np.zeros(3))
    ant = sum((o.antennas_offset for o in offsets), np.zeros(2))
    return MoveOffsets(pos, ori, ant)
