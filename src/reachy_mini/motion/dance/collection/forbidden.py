"""Forbidden and Experimental Moves, use with caution.

--------------------------------------
This module contains experimental and diagnostic moves that are not part of
the main, curated library. These moves may be unstable, highly specialized,
or intended for specific tests.

Use them with caution as these could damage the robot.
They are not included in the default `AVAILABLE_MOVES`
registry and must be imported explicitly.
"""

from dataclasses import replace
from typing import Any, Callable

import numpy as np

# Import all the necessary building blocks from the core motion library.
from reachy_mini.motion.collection.dance import DEFAULT_ANTENNA_PARAMS
from reachy_mini.motion.rhythmic_motion import (
    AVAILABLE_ANTENNA_MOVES,
    MoveOffsets,
    OscillationParams,
    atomic_roll,
    atomic_x_pos,
    atomic_y_pos,
    combine_offsets,
)


def move_critical_frequency_sweep(
    t_beats: float,
    roll_amplitude_rad: float,
    start_subcycles: float,
    end_subcycles: float,
    num_steps: int,
    step_duration_beats: float,
    antenna_amplitude_rad: float,
    antenna_move_name: str = "wiggle",
) -> MoveOffsets:
    """Perform a roll sweep across a frequency range to find critical points.

    This is a diagnostic move. It starts at a low frequency and increases
    in discrete steps to a high frequency, printing the current speed at each
    step. This helps identify speeds at which the physical robot may become
    unstable or fail to keep up.

    Args:
        t_beats (float): Continuous time in beats.
        roll_amplitude_rad (float): The amplitude of the side-to-side roll.
        start_subcycles (float): The starting frequency in subcycles/beat.
        end_subcycles (float): The ending frequency in subcycles/beat.
        num_steps (int): The number of discrete frequency steps in the sweep.
        step_duration_beats (float): How long to hold each frequency step, in beats.
        antenna_amplitude_rad (float): The amplitude of the antenna motion.
        antenna_move_name (str): The style of antenna motion.

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    # This state needs to persist across calls to this function.
    # A nonlocal variable is a clean way to achieve this without global scope pollution.
    if not hasattr(move_critical_frequency_sweep, "last_printed_step"):
        move_critical_frequency_sweep.last_printed_step = -1

    sweep_period_beats = num_steps * step_duration_beats
    t_in_sweep = t_beats % sweep_period_beats

    # Determine which step of the sweep we are currently in.
    current_step = int(np.floor(t_in_sweep / step_duration_beats))

    # Linearly interpolate to find the current frequency for this step.
    # We use (num_steps - 1) to ensure the final step reaches end_subcycles.
    if num_steps > 1:
        progression = current_step / (num_steps - 1)
        current_subcycles = (
            start_subcycles + (end_subcycles - start_subcycles) * progression
        )
    else:
        current_subcycles = start_subcycles

    # Print the current frequency ONLY when the step changes.
    if current_step != move_critical_frequency_sweep.last_printed_step:
        print(
            f"\n--- Sweep Step {current_step + 1}/{num_steps}: Freq = {current_subcycles:.2f} subcycles/beat ---"
        )
        move_critical_frequency_sweep.last_printed_step = current_step

    # Perform the roll motion at the calculated frequency.
    roll_params = OscillationParams(
        amplitude=roll_amplitude_rad,
        subcycles_per_beat=current_subcycles,
    )
    base = atomic_roll(t_beats, roll_params)

    antenna_params = OscillationParams(
        amplitude=antenna_amplitude_rad,
        subcycles_per_beat=current_subcycles,
    )
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)

    return combine_offsets([base, antennas])


def move_ellipse_walk(
    t_beats: float,
    x_amplitude_m: float,
    y_amplitude_m: float,
    subcycles_per_beat: float,
    antenna_amplitude_rad: float,
    antenna_move_name: str = "wiggle",
    phase_offset: float = 0.0,
    waveform: str = "sin",
) -> MoveOffsets:
    """Create a walking-like motion by tracing an ellipse in the X-Y plane.

    This move shifts the robot's weight in a smooth elliptical pattern, which
    can induce a shuffling or walking motion on some surfaces. Z remains constant.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        x_amplitude_m (float): The forward/backward radius of the ellipse in meters.
        y_amplitude_m (float): The side-to-side radius of the ellipse in meters.
        subcycles_per_beat (float): Number of movement oscillations per beat.
        antenna_amplitude_rad (float): The amplitude of the antenna motion in radians.
        antenna_move_name (str): The style of antenna motion (e.g. 'wiggle' or 'both').
        phase_offset (float): A normalized phase offset for the motion as a fraction of a cycle.
            0.5 shifts by half a period.
        waveform (str): The shape of the oscillation.

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    base_params = OscillationParams(0, subcycles_per_beat, phase_offset, waveform)

    # Create the forward/backward (X) component of the motion
    x_params = replace(base_params, amplitude=x_amplitude_m)
    x_motion = atomic_x_pos(t_beats, x_params)

    # Create the side-to-side (Y) component, shifted by a quarter phase for an ellipse
    y_params = replace(
        base_params,
        amplitude=y_amplitude_m,
        phase_offset=base_params.phase_offset + 0.25,
    )
    y_motion = atomic_y_pos(t_beats, y_params)

    # Antennas can follow the main rhythm
    antenna_params = OscillationParams(
        antenna_amplitude_rad, subcycles_per_beat, phase_offset, waveform
    )
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)

    return combine_offsets([x_motion, y_motion, antennas])


def move_crescent_walk(
    t_beats: float,
    x_amplitude_m: float,
    y_amplitude_m: float,
    subcycles_per_beat: float,
    antenna_amplitude_rad: float,
    antenna_move_name: str = "wiggle",
    phase_offset: float = 0.0,
) -> MoveOffsets:
    """Create a walking motion by tracing a crescent or 'M' shape.

    This move creates a path that pushes forward in X during both the
    left and right phases of the Y-axis oscillation. This can produce a
    more aggressive forward shuffle than a simple ellipse.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        x_amplitude_m (float): The maximum forward (X-axis) thrust in meters.
        y_amplitude_m (float): The side-to-side (Y-axis) amplitude in meters.
        subcycles_per_beat (float): Number of movement oscillations per beat.
        antenna_amplitude_rad (float): The amplitude of the antenna motion in radians.
        antenna_move_name (str): The style of antenna motion (e.g. 'wiggle' or 'both').
        phase_offset (float): A normalized phase offset for the motion as a fraction of a cycle.
            0.5 shifts by half a period.

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    # The phase determines the position within the crescent cycle.
    phase = 2 * np.pi * (subcycles_per_beat * t_beats + phase_offset)

    # Parametric generation of the crescent shape
    # Y follows a standard cosine wave for side-to-side motion.
    y_offset = -y_amplitude_m * np.cos(phase)
    # X follows the absolute value of a sine wave, pushing forward twice per cycle.
    x_offset = x_amplitude_m * np.abs(np.sin(phase))

    # This is a custom path, so we build the base MoveOffsets directly.
    base_move = MoveOffsets(np.array([x_offset, y_offset, 0]), np.zeros(3), np.zeros(2))

    # Antennas can still follow a simple oscillation.
    antenna_params = OscillationParams(
        antenna_amplitude_rad, subcycles_per_beat, phase_offset
    )
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)

    return combine_offsets([base_move, antennas])


# A separate dictionary for forbidden moves, allowing them to be imported explicitly.
FORBIDDEN_MOVES: dict[str, tuple[Callable[..., MoveOffsets], dict[str, Any]]] = {
    "critical_frequency_sweep": (
        move_critical_frequency_sweep,
        {
            "roll_amplitude_rad": np.deg2rad(40),
            "start_subcycles": 0.5,
            "end_subcycles": 4.0,
            "num_steps": 10,
            "step_duration_beats": 16,
            **DEFAULT_ANTENNA_PARAMS,
        },
    ),
    "ellipse_walk": (
        move_ellipse_walk,
        {
            "x_amplitude_m": 0.01,
            "y_amplitude_m": 0.025,
            "subcycles_per_beat": 1.55,
            **DEFAULT_ANTENNA_PARAMS,
        },
    ),
    "crescent_walk": (
        move_crescent_walk,
        {
            "x_amplitude_m": 0.03,
            "y_amplitude_m": 0.03,
            "subcycles_per_beat": 1.55,
            **DEFAULT_ANTENNA_PARAMS,
        },
    ),
}
