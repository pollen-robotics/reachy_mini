"""Utility functions for motion generation in Reachy Mini.

A rich, compositional library for creating life-like, expressive, and rhythmic
motion for robotic characters.
This module provides functions to generate oscillatory and transient motions,
as well as a utility to combine multiple motion offsets into a single offset.

Core Philosophy
---------------
This library is built on a compositional pattern:
1.  **Motion Primitives:** Low-level functions like `oscillation_motion` (for
    continuous loops) and `transient_motion` (for one-shot, eased actions)
    form the mathematical foundation.
2.  **Atomic Moves:** Simple wrappers around primitives that control a single
    axis or function (e.g., `atomic_pitch`, `atomic_antenna_wiggle`). These
    are the fundamental building blocks.
3.  **Choreographed Moves:** The main library of named moves, created by
    combining multiple atomic moves using the `combine_offsets` utility.

"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MoveOffsets:
    """Data structure to hold motion offsets for position, orientation, and antennas."""

    position_offset: np.ndarray  # Shape: (3,) - x, y, z in meters
    orientation_offset: np.ndarray  # Shape: (3,) - roll, pitch, yaw in radians
    antennas_offset: np.ndarray  # Shape: (2,) - left, right in radians


def oscillation_motion(
    t_beats: float,
    amplitude: float,
    subcycles_per_beat: float = 1.0,
    phase_offset: float = 0.0,
    waveform: str = "sin",
) -> float:
    """Generate an oscillatory motion based on the specified parameters.

    Args:
        t_beats (float): Time in beats at which to evaluate the motion (this is expected to increase by 1 one each beat). Beware that this is not the same as time in seconds but in beats (dimensionless).
        amplitude (float): Maximum amplitude of the oscillation.
        subcycles_per_beat (float): Number of oscillation subcycles per beat.
        phase_offset (float): Phase offset in cycles, to shift the waveform.
        waveform (str): Type of waveform to generate ('sin', 'cos', 'square', 'triangle', 'sawtooth').

    Returns:
        float: The value of the oscillation at time `t_beats`.

    """
    phase = 2 * np.pi * (subcycles_per_beat * t_beats + phase_offset)
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
) -> float:
    """Generate a transient motion that eases in and out over a specified duration.

    Args:
        t_beats (float): Time in beats at which to evaluate the motion (this is expected to increase by 1 one each beat). Beware that this is not the same as time in seconds but in beats (dimensionless).
        amplitude (float): Maximum amplitude of the transient motion.
        duration_beats (float): Duration of the transient motion in beats.
        delay_beats (float): Delay before the transient motion starts, in beats.
        repeat_every (float): If greater than 0, the transient motion will repeat every `repeat_every` beats. If 0, it will not repeat.

    Returns:
        float: The value of the transient motion at time `t_beats`.

    """
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
    """Combine multiple MoveOffsets into a single MoveOffsets.

    Args:
        *offsets (MoveOffsets): Variable number of MoveOffsets to combine.

    Returns:
        MoveOffsets: A new MoveOffsets instance with combined position, orientation, and antennas offsets.

    """
    if not offsets:
        return MoveOffsets(np.zeros(3), np.zeros(3), np.zeros(2))

    pos = sum([o.position_offset for o in offsets], np.zeros(3))
    ori = sum([o.orientation_offset for o in offsets], np.zeros(3))
    ant = sum([o.antennas_offset for o in offsets], np.zeros(2))

    return MoveOffsets(pos, ori, ant)
