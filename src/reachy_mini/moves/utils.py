"""Utility functions for motion generation in Reachy Mini.

A compositional library for creating rhythmic
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
from typing import List
import numpy as np


@dataclass
class MoveOffsets:
    """Data structure to hold motion offsets for position, orientation, and antennas."""

    position_offset: np.ndarray  # Shape: (3,) - x, y, z in meters
    orientation_offset: np.ndarray  # Shape: (3,) - roll, pitch, yaw in radians
    antennas_offset: np.ndarray  # Shape: (2,) - left, right in radians


@dataclass
class OscillationParams:
    """Parameters for oscillation motion."""

    amplitude: float  # float: Maximum amplitude of the oscillation.
    subcycles_per_beat: float = 1.0  # float: Number of oscillation subcycles per beat.
    phase_offset: float = 0.0  # float: Phase offset in cycles, to shift the waveform.
    waveform: str = "sin"  # str: Type of waveform to generate ('sin', 'cos', 'square', 'triangle', 'sawtooth').

@dataclass
class TransientParams:
    """Define parameters for a one-shot, transient motion."""

    amplitude: float  # Peak value of the motion, in radians or meters.
    duration_in_beats: float = 1.0  # The time, in beats, over which the motion occurs.
    delay_beats: float = 0.0  # An initial delay, in beats, before the motion starts.
    repeat_every: float = 0.0  # If > 0, repeat motion at this interval. If 0, it's a one-shot move.


def oscillation_motion(
    t_beats: float,
    params: OscillationParams,
) -> float:
    """Generate an oscillatory motion based on the specified parameters.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion, increases by 1 every beat. t_beats [dimensionless] = time_in_seconds [seconds] * frequency [hertz].
        params (OscillationParams): Parameters for the oscillation motion.

    Returns:
        float: The value of the oscillation at time `t_beats`.
        
    Raises:
        ValueError: If the `waveform` in `params` is not a supported type.

    """
    phase = 2 * np.pi * (params.subcycles_per_beat * t_beats + params.phase_offset)

    if params.waveform == "sin":
        return params.amplitude * np.sin(phase)
    elif params.waveform == "cos":
        return params.amplitude * np.cos(phase)
    elif params.waveform == "square":
        return params.amplitude * np.sign(np.sin(phase))
    elif params.waveform == "triangle":
        return params.amplitude * (2 / np.pi) * np.arcsin(np.sin(phase))
    elif params.waveform == "sawtooth":
        return params.amplitude * (2 * ((phase / (2 * np.pi)) % 1) - 1)

    raise ValueError(f"Unsupported waveform type: {params.waveform}")

def transient_motion(t_beats: float, params: TransientParams) -> float:
    """Generate a single, eased motion that occurs over a specific duration.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        params (TransientParams): An object containing parameters for the transient motion.

    Returns:
        float: The calculated value of the motion at the given time.

    """
    # If repeat_every is specified, use the repeating logic.
    if params.repeat_every > 0.0:
        start_time = (
            np.floor((t_beats - params.delay_beats) / params.repeat_every) * params.repeat_every
            + params.delay_beats
        )
    # Otherwise, it's a true one-shot move. Don't calculate a repeating start_time.
    else:
        start_time = params.delay_beats

    if start_time <= t_beats < start_time + params.duration_in_beats:
        t_norm = (t_beats - start_time) / params.duration_in_beats
        # Apply a "smoothstep" easing function: 3t^2 - 2t^3
        eased_t = t_norm * t_norm * (3.0 - 2.0 * t_norm)
        return params.amplitude * eased_t

    return 0.0

def combine_offsets(offsets_list: List[MoveOffsets]) -> MoveOffsets:
    """Combine multiple MoveOffsets into a single object by summing them.

    Args:
        offsets_list (List[MoveOffsets]): A list of MoveOffsets objects to combine.

    Returns:
        MoveOffsets: A new MoveOffsets instance with the summed offsets.

    """
    if not offsets_list:
        return MoveOffsets(np.zeros(3), np.zeros(3), np.zeros(2))

    pos = sum([o.position_offset for o in offsets_list], np.zeros(3))
    ori = sum([o.orientation_offset for o in offsets_list], np.zeros(3))
    ant = sum([o.antennas_offset for o in offsets_list], np.zeros(2))

    return MoveOffsets(pos, ori, ant)


def _test_transient_motion():
    """Run a few examples of transient_motion and print the output."""
    print("=" * 50)
    print(" DEMONSTRATION OF transient_motion() ".center(50, "="))
    print("=" * 50)

    # --- Case 1: A simple one-shot motion ---
    print("\n--- Case 1: Simple One-Shot (duration=2.0) ---")
    print("The motion should start at t=0.0, hit its peak at t=2.0, and then stay there.")
    params1 = TransientParams(amplitude=10.0, duration_in_beats=2.0)
    for t in np.arange(0, 4.25, 0.25):
        value = transient_motion(t, params1)
        print(f"t={t:4.2f} -> value={value:6.3f}")

    # --- Case 2: A one-shot motion with a delay ---
    print("\n--- Case 2: One-Shot with Delay (duration=2.0, delay=1.0) ---")
    print("The motion should be 0 until t=1.0, then start, and finish at t=3.0.")
    params2 = TransientParams(amplitude=10.0, duration_in_beats=2.0, delay_beats=1.0)
    for t in np.arange(0, 5.25, 0.25):
        value = transient_motion(t, params2)
        print(f"t={t:4.2f} -> value={value:6.3f}")

    # --- Case 3: A repeating motion ---
    print("\n--- Case 3: Repeating Motion (duration=1.0, repeat_every=4.0) ---")
    print("A short 1-beat motion will occur every 4 beats.")
    print("Look for motion during [0,1), [4,5), and [8,9).")
    params3 = TransientParams(amplitude=5.0, duration_in_beats=1.0, repeat_every=4.0)
    for t in np.arange(0, 10.25, 0.25):
        value = transient_motion(t, params3)
        print(f"t={t:4.2f} -> value={value:6.3f}")

    print("\n" + "=" * 50)
    
if __name__ == "__main__":
    _test_transient_motion()