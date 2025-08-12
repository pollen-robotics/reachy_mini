"""Dance Moves Collection for Reachy Mini.

----------------------------------
This module provides a library of high-level, choreographed dance moves.
It uses the primitives and atomic building blocks from the `rhythmic_motion`
module to create complex, named, tunable dance moves.

The main entry point is the `AVAILABLE_MOVES` dictionary, which maps move
names to their respective functions and default parameters.
"""

from dataclasses import replace
from typing import Any, Callable

import numpy as np

# Import all the necessary building blocks from the core motion library.
from ..rhythmic_motion import (
    AVAILABLE_ANTENNA_MOVES,
    MoveOffsets,
    OscillationParams,
    TransientParams,
    atomic_pitch,
    atomic_roll,
    atomic_x_pos,
    atomic_y_pos,
    atomic_yaw,
    atomic_z_pos,
    combine_offsets,
    transient_motion,
)


def move_simple_nod(
    t_beats: float,
    amplitude_rad: float,
    antenna_amplitude_rad: float,
    subcycles_per_beat: float,
    antenna_move_name: str = "wiggle",
    phase_offset: float = 0.0,
    waveform: str = "sin",
) -> MoveOffsets:
    """Generate a simple, continuous nodding motion.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        amplitude_rad (float): The primary amplitude of the nod in radians.
        antenna_amplitude_rad (float): The amplitude of the antenna motion in radians.
        subcycles_per_beat (float): Number of movement oscillations per beat.
        antenna_move_name (str): The style of antenna motion (e.g. 'wiggle' or 'both').
        phase_offset (float): A normalized phase offset for the motion as a fraction of a cycle.
            0.5 shifts by half a period.
        waveform (str): The shape of the oscillation.

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    base_params = OscillationParams(
        amplitude_rad, subcycles_per_beat, phase_offset, waveform
    )
    antenna_params = OscillationParams(
        antenna_amplitude_rad, subcycles_per_beat, phase_offset, waveform
    )
    base = atomic_pitch(t_beats, base_params)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)
    return combine_offsets([base, antennas])


def move_head_tilt_roll(
    t_beats: float,
    amplitude_rad: float,
    antenna_amplitude_rad: float,
    subcycles_per_beat: float,
    antenna_move_name: str = "wiggle",
    phase_offset: float = 0.0,
    waveform: str = "sin",
) -> MoveOffsets:
    """Generate a continuous side-to-side head roll.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        amplitude_rad (float): The primary amplitude of the roll in radians.
        antenna_amplitude_rad (float): The amplitude of the antenna motion in radians.
        subcycles_per_beat (float): Number of movement oscillations per beat.
        antenna_move_name (str): The style of antenna motion (e.g. 'wiggle' or 'both').
        phase_offset (float): A normalized phase offset for the motion as a fraction of a cycle.
            0.5 shifts by half a period.
        waveform (str): The shape of the oscillation.

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    base_params = OscillationParams(
        amplitude_rad, subcycles_per_beat, phase_offset, waveform
    )
    antenna_params = OscillationParams(
        antenna_amplitude_rad, subcycles_per_beat, phase_offset, waveform
    )
    base = atomic_roll(t_beats, base_params)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)
    return combine_offsets([base, antennas])


def move_side_to_side_sway(
    t_beats: float,
    amplitude_m: float,
    antenna_amplitude_rad: float,
    subcycles_per_beat: float,
    antenna_move_name: str = "wiggle",
    phase_offset: float = 0.0,
    waveform: str = "sin",
) -> MoveOffsets:
    """Generate a continuous, side-to-side sway of the body.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        amplitude_m (float): The primary amplitude of the sway in meters.
        antenna_amplitude_rad (float): The amplitude of the antenna motion in radians.
        subcycles_per_beat (float): Number of movement oscillations per beat.
        antenna_move_name (str): The style of antenna motion (e.g. 'wiggle' or 'both').
        phase_offset (float): A normalized phase offset for the motion as a fraction of a cycle.
            0.5 shifts by half a period.
        waveform (str): The shape of the oscillation.

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    base_params = OscillationParams(
        amplitude_m, subcycles_per_beat, phase_offset, waveform
    )
    antenna_params = OscillationParams(
        antenna_amplitude_rad, subcycles_per_beat, phase_offset, waveform
    )
    base = atomic_y_pos(t_beats, base_params)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)
    return combine_offsets([base, antennas])


def move_dizzy_spin(
    t_beats: float,
    roll_amplitude_rad: float,
    pitch_amplitude_rad: float,
    antenna_amplitude_rad: float,
    subcycles_per_beat: float,
    antenna_move_name: str = "wiggle",
    phase_offset: float = 0.0,
    waveform: str = "sin",
) -> MoveOffsets:
    """Create a circular, dizzying head motion by combining roll and pitch.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        roll_amplitude_rad (float): The amplitude of the roll component.
        pitch_amplitude_rad (float): The amplitude of the pitch component.
        antenna_amplitude_rad (float): The amplitude of the antenna motion in radians.
        subcycles_per_beat (float): Number of movement oscillations per beat.
        antenna_move_name (str): The style of antenna motion (e.g. 'wiggle' or 'both').
        phase_offset (float): A normalized phase offset for the motion as a fraction of a cycle.
            0.5 shifts by half a period.
        waveform (str): The shape of the oscillation.

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    base_params = OscillationParams(
        0, subcycles_per_beat, phase_offset, waveform
    )  # Amplitude is placeholder
    roll_params = replace(base_params, amplitude=roll_amplitude_rad)
    pitch_params = replace(
        base_params,
        amplitude=pitch_amplitude_rad,
        phase_offset=base_params.phase_offset + 0.25,
    )
    antenna_params = replace(base_params, amplitude=antenna_amplitude_rad)
    roll = atomic_roll(t_beats, roll_params)
    pitch = atomic_pitch(t_beats, pitch_params)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)
    return combine_offsets([roll, pitch, antennas])


def move_stumble_and_recover(
    t_beats: float,
    yaw_amplitude_rad: float,
    pitch_amplitude_rad: float,
    y_amplitude_m: float,
    antenna_amplitude_rad: float,
    subcycles_per_beat: float,
    antenna_move_name: str = "both",
    phase_offset: float = 0.0,
    waveform: str = "sin",
) -> MoveOffsets:
    """Simulate a stumble and recovery with yaw, fast pitch, and stabilizing sway.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        yaw_amplitude_rad (float): The amplitude of the main yaw stumble.
        pitch_amplitude_rad (float): The amplitude of the faster pitch correction.
        y_amplitude_m (float): The amplitude of the stabilizing side sway.
        antenna_amplitude_rad (float): The amplitude of the antenna motion in radians.
        subcycles_per_beat (float): Number of movement oscillations per beat.
        antenna_move_name (str): The style of antenna motion (e.g. 'wiggle' or 'both').
        phase_offset (float): A normalized phase offset for the motion as a fraction of a cycle.
            0.5 shifts by half a period.
        waveform (str): The shape of the oscillation.

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    base_params = OscillationParams(0, subcycles_per_beat, phase_offset, waveform)
    yaw_params = replace(base_params, amplitude=yaw_amplitude_rad)
    pitch_params = replace(
        base_params,
        amplitude=pitch_amplitude_rad,
        subcycles_per_beat=base_params.subcycles_per_beat * 2,
    )
    sway_params = replace(
        base_params,
        amplitude=y_amplitude_m,
        phase_offset=base_params.phase_offset + 0.5,
    )
    antenna_params = replace(base_params, amplitude=antenna_amplitude_rad)
    yaw = atomic_yaw(t_beats, yaw_params)
    pitch = atomic_pitch(t_beats, pitch_params)
    stabilizer_sway = atomic_y_pos(t_beats, sway_params)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)
    return combine_offsets([yaw, pitch, stabilizer_sway, antennas])


def move_headbanger_combo(
    t_beats: float,
    pitch_amplitude_rad: float,
    z_amplitude_m: float,
    antenna_amplitude_rad: float,
    subcycles_per_beat: float,
    antenna_move_name: str = "both",
    phase_offset: float = 0.0,
    waveform: str = "sin",
) -> MoveOffsets:
    """Combine a strong pitch nod with a vertical bounce for a headbanging effect.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        pitch_amplitude_rad (float): The amplitude of the primary head nod.
        z_amplitude_m (float): The amplitude of the vertical body bounce.
        antenna_amplitude_rad (float): The amplitude of the antenna motion in radians.
        subcycles_per_beat (float): Number of movement oscillations per beat.
        antenna_move_name (str): The style of antenna motion (e.g. 'wiggle' or 'both').
        phase_offset (float): A normalized phase offset for the motion as a fraction of a cycle.
            0.5 shifts by half a period.
        waveform (str): The shape of the oscillation.

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    base_params = OscillationParams(0, subcycles_per_beat, phase_offset, waveform)
    nod_params = replace(base_params, amplitude=pitch_amplitude_rad)
    bounce_params = replace(
        base_params,
        amplitude=z_amplitude_m,
        phase_offset=base_params.phase_offset + 0.1,
    )
    antenna_params = replace(base_params, amplitude=antenna_amplitude_rad)
    nod = atomic_pitch(t_beats, nod_params)
    bounce = atomic_z_pos(t_beats, bounce_params)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)
    return combine_offsets([nod, bounce, antennas])


def move_interwoven_spirals(
    t_beats: float,
    roll_amp: float,
    pitch_amp: float,
    yaw_amp: float,
    antenna_amplitude_rad: float,
    subcycles_per_beat: float,
    antenna_move_name: str = "wiggle",
    phase_offset: float = 0.0,
    waveform: str = "sin",
) -> MoveOffsets:
    """Create a complex spiral motion by combining axes at different frequencies.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        roll_amp (float): The amplitude of the roll component.
        pitch_amp (float): The amplitude of the pitch component.
        yaw_amp (float): The amplitude of the yaw component.
        antenna_amplitude_rad (float): The amplitude of the antenna motion in radians.
        subcycles_per_beat (float): Number of movement oscillations per beat (for antennas).
        antenna_move_name (str): The style of antenna motion (e.g. 'wiggle' or 'both').
        phase_offset (float): A normalized phase offset for the motion as a fraction of a cycle.
            0.5 shifts by half a period.
        waveform (str): The shape of the oscillation.

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    base_params = OscillationParams(0, subcycles_per_beat, phase_offset, waveform)
    roll_params = replace(base_params, amplitude=roll_amp, subcycles_per_beat=0.125)
    pitch_params = replace(
        base_params,
        amplitude=pitch_amp,
        subcycles_per_beat=0.25,
        phase_offset=base_params.phase_offset + 0.25,
    )
    yaw_params = replace(
        base_params,
        amplitude=yaw_amp,
        subcycles_per_beat=0.5,
        phase_offset=base_params.phase_offset + 0.5,
    )
    antenna_params = replace(base_params, amplitude=antenna_amplitude_rad)
    roll = atomic_roll(t_beats, roll_params)
    pitch = atomic_pitch(t_beats, pitch_params)
    yaw = atomic_yaw(t_beats, yaw_params)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)
    return combine_offsets([roll, pitch, yaw, antennas])


def move_sharp_side_tilt(
    t_beats: float,
    roll_amplitude_rad: float,
    antenna_amplitude_rad: float,
    subcycles_per_beat: float,
    antenna_move_name: str = "wiggle",
    phase_offset: float = 0.0,
    waveform: str = "triangle",
) -> MoveOffsets:
    """Perform a sharp, quick side-to-side tilt using a triangle waveform.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        roll_amplitude_rad (float): The primary amplitude of the tilt in radians.
        antenna_amplitude_rad (float): The amplitude of the antenna motion in radians.
        subcycles_per_beat (float): Number of movement oscillations per beat.
        antenna_move_name (str): The style of antenna motion (e.g. 'wiggle' or 'both').
        phase_offset (float): A normalized phase offset for the motion as a fraction of a cycle.
            0.5 shifts by half a period.
        waveform (str): The shape of the oscillation (defaults to 'triangle').

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    base_params = OscillationParams(
        roll_amplitude_rad, subcycles_per_beat, phase_offset, waveform
    )
    antenna_params = OscillationParams(
        antenna_amplitude_rad, subcycles_per_beat, phase_offset, waveform
    )
    base = atomic_roll(t_beats, base_params)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)
    return combine_offsets([base, antennas])


def move_side_peekaboo(
    t_beats: float,
    z_amp: float,
    y_amp: float,
    pitch_amp: float,
    antenna_amplitude_rad: float,
    antenna_move_name: str = "both",
    subcycles_per_beat: float = 0.5,
) -> MoveOffsets:
    """Perform a complete peekaboo 'performance' with a start, middle, and end.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        z_amp (float): The amplitude of the vertical (hide/unhide) motion in meters.
        y_amp (float): The amplitude of the horizontal (peek) motion in meters.
        pitch_amp (float): The amplitude of the 'excited nod' in radians.
        antenna_amplitude_rad (float): The amplitude of the antenna motion in radians.
        antenna_move_name (str): The style of antenna motion (e.g. 'wiggle' or 'both').
        subcycles_per_beat (float): Number of movement oscillations per beat (for antennas).

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    period = 10.0
    t_in_period = t_beats % period
    pos, ori = np.zeros(3), np.zeros(3)

    def ease(t: float) -> float:
        t_clipped = np.clip(t, 0.0, 1.0)
        return t_clipped * t_clipped * (3 - 2 * t_clipped)

    def excited_nod(t: float) -> float:
        return pitch_amp * np.sin(np.clip(t, 0.0, 1.0) * np.pi)

    if t_in_period < 1.0:
        t = t_in_period / 1.0
        pos[2] = -z_amp * ease(t)
    elif t_in_period < 3.0:
        t = (t_in_period - 1.0) / 2.0
        pos[2] = -z_amp * (1 - ease(t))
        pos[1] = y_amp * ease(t)
        ori[1] = excited_nod(t)
    elif t_in_period < 5.0:
        t = (t_in_period - 3.0) / 2.0
        pos[2] = -z_amp * ease(t)
        pos[1] = y_amp * (1 - ease(t))
    elif t_in_period < 7.0:
        t = (t_in_period - 5.0) / 2.0
        pos[2] = -z_amp * (1 - ease(t))
        pos[1] = -y_amp * ease(t)
        ori[1] = -excited_nod(t)
    elif t_in_period < 9.0:
        t = (t_in_period - 7.0) / 2.0
        pos[2] = -z_amp * ease(t)
        pos[1] = -y_amp * (1 - ease(t))
    else:
        t = (t_in_period - 9.0) / 1.0
        pos[2] = -z_amp * (1 - ease(t))
    antenna_params = OscillationParams(antenna_amplitude_rad, subcycles_per_beat)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)
    return combine_offsets([MoveOffsets(pos, ori, np.zeros(2)), antennas])


def move_yeah_nod(
    t_beats: float,
    amplitude_rad: float,
    antenna_amplitude_rad: float,
    subcycles_per_beat: float,
    antenna_move_name: str = "both",
) -> MoveOffsets:
    """Generate an emphatic two-part 'yeah' nod using transient motions.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        amplitude_rad (float): The primary amplitude of the main nod.
        antenna_amplitude_rad (float): The amplitude of the antenna motion in radians.
        subcycles_per_beat (float): Number of movement oscillations per beat.
        antenna_move_name (str): The style of antenna motion (e.g. 'wiggle' or 'both').

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    repeat_every = 1.0 / subcycles_per_beat
    nod1_params = TransientParams(
        amplitude_rad, duration_in_beats=repeat_every * 0.4, repeat_every=repeat_every
    )
    nod2_params = TransientParams(
        amplitude_rad * 0.7,
        duration_in_beats=repeat_every * 0.3,
        delay_beats=repeat_every * 0.5,
        repeat_every=repeat_every,
    )
    nod1 = transient_motion(t_beats, nod1_params)
    nod2 = transient_motion(t_beats, nod2_params)
    base = MoveOffsets(np.zeros(3), np.array([0, nod1 + nod2, 0]), np.zeros(2))
    antenna_params = OscillationParams(antenna_amplitude_rad, subcycles_per_beat)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)
    return combine_offsets([base, antennas])


def move_uh_huh_tilt(
    t_beats: float,
    amplitude_rad: float,
    antenna_amplitude_rad: float,
    subcycles_per_beat: float,
    antenna_move_name: str = "wiggle",
    phase_offset: float = 0.0,
    waveform: str = "sin",
) -> MoveOffsets:
    """Create a combined roll-and-pitch 'uh-huh' gesture.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        amplitude_rad (float): The primary amplitude for both roll and pitch.
        antenna_amplitude_rad (float): The amplitude of the antenna motion in radians.
        subcycles_per_beat (float): Number of movement oscillations per beat.
        antenna_move_name (str): The style of antenna motion (e.g. 'wiggle' or 'both').
        phase_offset (float): A normalized phase offset for the motion as a fraction of a cycle.
            0.5 shifts by half a period.
        waveform (str): The shape of the oscillation.

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    base_params = OscillationParams(
        amplitude_rad, subcycles_per_beat, phase_offset, waveform
    )
    antenna_params = OscillationParams(
        antenna_amplitude_rad, subcycles_per_beat, phase_offset, waveform
    )
    roll = atomic_roll(t_beats, base_params)
    pitch = atomic_pitch(t_beats, base_params)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)
    return combine_offsets([roll, pitch, antennas])


def move_neck_recoil(
    t_beats: float,
    amplitude_m: float,
    antenna_amplitude_rad: float,
    subcycles_per_beat: float,
    antenna_move_name: str = "wiggle",
) -> MoveOffsets:
    """Simulate a quick, transient backward recoil of the neck.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        amplitude_m (float): The amplitude of the recoil in meters.
        antenna_amplitude_rad (float): The amplitude of the antenna motion in radians.
        subcycles_per_beat (float): Number of movement oscillations per beat.
        antenna_move_name (str): The style of antenna motion (e.g. 'wiggle' or 'both').

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    repeat_every = 1.0 / subcycles_per_beat
    recoil_params = TransientParams(
        -amplitude_m, duration_in_beats=repeat_every * 0.3, repeat_every=repeat_every
    )
    recoil = transient_motion(t_beats, recoil_params)
    base = MoveOffsets(np.array([recoil, 0, 0]), np.zeros(3), np.zeros(2))
    antenna_params = OscillationParams(antenna_amplitude_rad, subcycles_per_beat)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)
    return combine_offsets([base, antennas])


def move_chin_lead(
    t_beats: float,
    x_amplitude_m: float,
    pitch_amplitude_rad: float,
    antenna_amplitude_rad: float,
    subcycles_per_beat: float,
    antenna_move_name: str = "both",
    phase_offset: float = 0.0,
    waveform: str = "sin",
) -> MoveOffsets:
    """Create a forward motion led by the chin, combining x-translation and pitch.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        x_amplitude_m (float): The amplitude of the forward (X-axis) motion.
        pitch_amplitude_rad (float): The amplitude of the accompanying pitch.
        antenna_amplitude_rad (float): The amplitude of the antenna motion in radians.
        subcycles_per_beat (float): Number of movement oscillations per beat.
        antenna_move_name (str): The style of antenna motion (e.g. 'wiggle' or 'both').
        phase_offset (float): A normalized phase offset for the motion as a fraction of a cycle.
            0.5 shifts by half a period.
        waveform (str): The shape of the oscillation.

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    base_params = OscillationParams(0, subcycles_per_beat, phase_offset, waveform)
    x_move_params = replace(base_params, amplitude=x_amplitude_m)
    pitch_move_params = replace(
        base_params,
        amplitude=pitch_amplitude_rad,
        phase_offset=base_params.phase_offset - 0.25,
    )
    antenna_params = replace(base_params, amplitude=antenna_amplitude_rad)
    x_move = atomic_x_pos(t_beats, x_move_params)
    pitch_move = atomic_pitch(t_beats, pitch_move_params)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)
    return combine_offsets([x_move, pitch_move, antennas])


def move_groovy_sway_and_roll(
    t_beats: float,
    y_amplitude_m: float,
    roll_amplitude_rad: float,
    antenna_amplitude_rad: float,
    subcycles_per_beat: float,
    antenna_move_name: str = "wiggle",
    phase_offset: float = 0.0,
    waveform: str = "sin",
) -> MoveOffsets:
    """Combine a side-to-side sway with a corresponding roll for a groovy effect.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        y_amplitude_m (float): The amplitude of the side-to-side (Y-axis) sway.
        roll_amplitude_rad (float): The amplitude of the accompanying roll.
        antenna_amplitude_rad (float): The amplitude of the antenna motion in radians.
        subcycles_per_beat (float): Number of movement oscillations per beat.
        antenna_move_name (str): The style of antenna motion (e.g. 'wiggle' or 'both').
        phase_offset (float): A normalized phase offset for the motion as a fraction of a cycle.
            0.5 shifts by half a period.
        waveform (str): The shape of the oscillation.

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    base_params = OscillationParams(0, subcycles_per_beat, phase_offset, waveform)
    sway_params = replace(base_params, amplitude=y_amplitude_m)
    roll_params = replace(
        base_params,
        amplitude=roll_amplitude_rad,
        phase_offset=base_params.phase_offset + 0.25,
    )
    antenna_params = replace(base_params, amplitude=antenna_amplitude_rad)
    sway = atomic_y_pos(t_beats, sway_params)
    roll = atomic_roll(t_beats, roll_params)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)
    return combine_offsets([sway, roll, antennas])


def move_chicken_peck(
    t_beats: float,
    amplitude_m: float,
    antenna_amplitude_rad: float,
    subcycles_per_beat: float,
    antenna_move_name: str = "both",
) -> MoveOffsets:
    """Simulate a chicken-like pecking motion.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        amplitude_m (float): The base amplitude for the forward peck.
        antenna_amplitude_rad (float): The amplitude of the antenna motion in radians.
        subcycles_per_beat (float): Number of movement oscillations per beat.
        antenna_move_name (str): The style of antenna motion (e.g. 'wiggle' or 'both').

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    repeat_every = 1.0 / subcycles_per_beat
    x_offset = transient_motion(
        t_beats,
        TransientParams(
            amplitude_m, duration_in_beats=repeat_every * 0.8, repeat_every=repeat_every
        ),
    )
    pitch_offset = transient_motion(
        t_beats,
        TransientParams(
            amplitude_m * 5,
            duration_in_beats=repeat_every * 0.8,
            repeat_every=repeat_every,
        ),
    )
    base = MoveOffsets(
        np.array([x_offset, 0, 0]), np.array([0, pitch_offset, 0]), np.zeros(2)
    )
    antenna_params = OscillationParams(antenna_amplitude_rad, subcycles_per_beat)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)
    return combine_offsets([base, antennas])


def move_side_glance_flick(
    t_beats: float,
    yaw_amplitude_rad: float,
    antenna_amplitude_rad: float,
    subcycles_per_beat: float,
    antenna_move_name: str = "wiggle",
) -> MoveOffsets:
    """Perform a quick, sharp glance to the side that holds and then returns.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        yaw_amplitude_rad (float): The amplitude of the glance in radians.
        antenna_amplitude_rad (float): The amplitude of the antenna motion in radians.
        subcycles_per_beat (float): Number of movement oscillations per beat.
        antenna_move_name (str): The style of antenna motion (e.g. 'wiggle' or 'both').

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    period = 1.0 / subcycles_per_beat
    t_in_period = t_beats % period

    def ease(t: float) -> float:
        return t * t * (3 - 2 * t)

    yaw_offset = 0
    if t_in_period < 0.125 * period:
        yaw_offset = yaw_amplitude_rad * ease(t_in_period / (0.125 * period))
    elif t_in_period < 0.375 * period:
        yaw_offset = yaw_amplitude_rad
    else:
        yaw_offset = yaw_amplitude_rad * (
            1.0 - ease((t_in_period - 0.375 * period) / (0.625 * period))
        )
    base = MoveOffsets(np.zeros(3), np.array([0, 0, yaw_offset]), np.zeros(2))
    antenna_params = OscillationParams(antenna_amplitude_rad, subcycles_per_beat)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)
    return combine_offsets([base, antennas])


def move_polyrhythm_combo(
    t_beats: float,
    sway_amplitude_m: float,
    nod_amplitude_rad: float,
    antenna_amplitude_rad: float,
    antenna_move_name: str = "wiggle",
) -> MoveOffsets:
    """Combine a 3-beat sway and a 2-beat nod to create a polyrhythmic feel.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        sway_amplitude_m (float): The amplitude of the 3-beat sway motion.
        nod_amplitude_rad (float): The amplitude of the 2-beat nod motion.
        antenna_amplitude_rad (float): The amplitude of the antenna motion in radians.
        antenna_move_name (str): The style of antenna motion (e.g. 'wiggle' or 'both').

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    sway_params = OscillationParams(sway_amplitude_m, subcycles_per_beat=1 / 3)
    nod_params = OscillationParams(nod_amplitude_rad, subcycles_per_beat=1 / 2)
    antenna_params = OscillationParams(
        antenna_amplitude_rad, subcycles_per_beat=1.0
    )  # Give antennas their own rhythm
    sway = atomic_y_pos(t_beats, sway_params)
    nod = atomic_pitch(t_beats, nod_params)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)
    return combine_offsets([sway, nod, antennas])


def move_grid_snap(
    t_beats: float,
    amplitude_rad: float,
    antenna_amplitude_rad: float,
    subcycles_per_beat: float,
    antenna_move_name: str = "both",
    phase_offset: float = 0.0,
) -> MoveOffsets:
    """Create a robotic motion that snaps to four corners and returns to center.

    This move defines a 5-step cycle: it snaps to four corners of a square
    in the yaw-pitch plane, and on the fifth step, it returns to the origin.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        amplitude_rad (float): The primary amplitude for both yaw and pitch,
            defining the corner positions in radians.
        antenna_amplitude_rad (float): The amplitude of the antenna motion in radians.
        subcycles_per_beat (float): The number of full 5-step snap cycles
            to perform per beat.
        antenna_move_name (str): The style of antenna motion (e.g. 'wiggle' or 'both').
        phase_offset (float): A normalized phase offset for the antenna motion as a
            fraction of a cycle. 0.5 shifts by half a period.

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    period = 1.0 / subcycles_per_beat
    t_in_period = t_beats % period

    # We define a 5-step cycle within the period: 4 corners, 1 return to center
    num_steps = 5.0
    step_duration = period / num_steps
    current_step = np.floor(t_in_period / step_duration)

    yaw_offset = 0.0
    pitch_offset = 0.0

    if current_step == 0:  # Top-right
        yaw_offset, pitch_offset = amplitude_rad, amplitude_rad
    elif current_step == 1:  # Bottom-right
        yaw_offset, pitch_offset = amplitude_rad, -amplitude_rad
    elif current_step == 2:  # Bottom-left
        yaw_offset, pitch_offset = -amplitude_rad, -amplitude_rad
    elif current_step == 3:  # Top-left
        yaw_offset, pitch_offset = -amplitude_rad, amplitude_rad
    else:  # Step 4: Return to center
        yaw_offset, pitch_offset = 0.0, 0.0

    base = MoveOffsets(
        np.zeros(3), np.array([0, pitch_offset, yaw_offset]), np.zeros(2)
    )
    antenna_params = OscillationParams(
        antenna_amplitude_rad, subcycles_per_beat, phase_offset
    )
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)
    return combine_offsets([base, antennas])


def move_pendulum_swing(
    t_beats: float,
    amplitude_rad: float,
    antenna_amplitude_rad: float,
    subcycles_per_beat: float,
    antenna_move_name: str = "wiggle",
    phase_offset: float = 0.0,
    waveform: str = "sin",
) -> MoveOffsets:
    """Simulate a simple, smooth pendulum swing using a roll motion.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        amplitude_rad (float): The primary amplitude of the swing in radians.
        antenna_amplitude_rad (float): The amplitude of the antenna motion in radians.
        subcycles_per_beat (float): Number of movement oscillations per beat.
        antenna_move_name (str): The style of antenna motion (e.g. 'wiggle' or 'both').
        phase_offset (float): A normalized phase offset for the motion as a fraction of a cycle.
            0.5 shifts by half a period.
        waveform (str): The shape of the oscillation.

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    base_params = OscillationParams(
        amplitude_rad, subcycles_per_beat, phase_offset, waveform
    )
    antenna_params = OscillationParams(
        antenna_amplitude_rad, subcycles_per_beat, phase_offset, waveform
    )
    base = atomic_roll(t_beats, base_params)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)
    return combine_offsets([base, antennas])


def move_jackson_square(
    t_beats: float,
    y_amp_m: float,
    z_amp_m: float,
    twitch_amplitude_rad: float,
    antenna_amplitude_rad: float,
    subcycles_per_beat: float,
    antenna_move_name: str = "wiggle",
    z_offset_m: float = -0.01,
) -> MoveOffsets:
    """Trace a vertically offset rectangle with twitches on arrival at each corner.

    This move follows a fixed 10-beat sequence composed of 5 "legs" of
    2-beats each. It starts from an offset origin, moves to the four corners
    of a rectangle, and returns to the start. A sharp, alternating roll twitch
    is executed upon arriving at each of the five path checkpoints.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion.
        y_amp_m (float): The half-width of the rectangle (Y-axis) in meters.
        z_amp_m (float): The half-height of the rectangle (Z-axis) in meters.
        twitch_amplitude_rad (float): The amplitude of the roll twitch in radians.
        antenna_amplitude_rad (float): The amplitude of the antenna motion in radians.
        subcycles_per_beat (float): Number of antenna oscillations per beat.
        antenna_move_name (str): The style of antenna motion (e.g. 'wiggle' or 'both').
        z_offset_m (float): A vertical offset in meters to lower or raise the rectangle.

    Returns:
        MoveOffsets: The calculated motion offsets.

    """
    period = 10.0  # 5 legs, 2 beats per leg
    leg_duration = 2.0
    t_in_period = t_beats % period
    pos, ori = np.zeros(3), np.zeros(3)

    # Define the 5 checkpoints of the path: Origin -> 4 corners
    path_points = np.array(
        [
            [0, 0, z_offset_m],  # P0: Offset Origin
            [0, y_amp_m, z_amp_m + z_offset_m],  # P1: Top-right
            [0, -y_amp_m, z_amp_m + z_offset_m],  # P2: Top-left
            [0, -y_amp_m, -z_amp_m + z_offset_m],  # P3: Bottom-left
            [0, y_amp_m, -z_amp_m + z_offset_m],  # P4: Bottom-right
        ]
    )

    def ease(t: float) -> float:
        t_clipped = np.clip(t, 0.0, 1.0)
        return t_clipped * t_clipped * (3 - 2 * t_clipped)

    # --- Corrected Twitch Logic ---
    # We want 5 twitches, one at the end of each 2-beat leg (t=2, 4, 6, 8, 10).
    # We achieve this by setting the delay to be almost a full leg duration.
    twitch_params = TransientParams(
        twitch_amplitude_rad,
        duration_in_beats=0.3,
        repeat_every=leg_duration,
        delay_beats=leg_duration - 0.15,  # This makes the first twitch start at t=1.85
    )
    twitch_val = transient_motion(t_beats, twitch_params)

    # Determine which leg of the journey we are on
    current_leg_index = int(np.floor(t_in_period / leg_duration))

    # The twitch direction alternates for each of the 5 arrival points.
    # We use `floor(t_beats / leg_duration)` to get a continuous index that
    # correctly identifies the twitch number (0 to 4) even across loops.
    twitch_number = int(np.floor(t_beats / leg_duration))
    twitch_direction = (-1) ** twitch_number
    ori[0] = twitch_val * twitch_direction

    # --- Path Interpolation Logic ---
    # Determine start and end points for the current leg
    start_point = path_points[current_leg_index]
    end_point = path_points[(current_leg_index + 1) % len(path_points)]

    # Calculate progress along the current leg and apply easing
    progress = (t_in_period % leg_duration) / leg_duration
    eased_progress = ease(progress)

    # Interpolate position between the start and end points
    pos = start_point + (end_point - start_point) * eased_progress

    base_move = MoveOffsets(pos, ori, np.zeros(2))
    antenna_params = OscillationParams(antenna_amplitude_rad, subcycles_per_beat)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)
    return combine_offsets([base_move, antennas])


# ────────────────────────── MASTER MOVE DICTIONARY ──────────────────────────
# A dictionary containing the default parameters for antenna motion.
DEFAULT_ANTENNA_PARAMS: dict[str, Any] = {
    "antenna_move_name": "wiggle",
    "antenna_amplitude_rad": np.deg2rad(45),
}

# This single dictionary is now the main entry point for all moves.
# It maps a move name to a tuple containing:
# 1. The callable move function.
# 2. A dictionary of its runtime parameters.
# 3. A dictionary of metadata (e.g., description, default duration).
AVAILABLE_MOVES: dict[
    str, tuple[Callable[..., MoveOffsets], dict[str, Any], dict[str, Any]]
] = {
    "simple_nod": (
        move_simple_nod,
        {
            "amplitude_rad": np.deg2rad(20),
            "subcycles_per_beat": 1.0,
            **DEFAULT_ANTENNA_PARAMS,
        },
        {
            "default_duration_beats": 4,
            "description": "A simple, continuous up-and-down nodding motion.",
        },
    ),
    "head_tilt_roll": (
        move_head_tilt_roll,
        {
            "amplitude_rad": np.deg2rad(15),
            "subcycles_per_beat": 0.5,
            **DEFAULT_ANTENNA_PARAMS,
        },
        {
            "default_duration_beats": 4,
            "description": "A continuous side-to-side head roll (ear to shoulder).",
        },
    ),
    "side_to_side_sway": (
        move_side_to_side_sway,
        {"amplitude_m": 0.04, "subcycles_per_beat": 0.5, **DEFAULT_ANTENNA_PARAMS},
        {
            "default_duration_beats": 4,
            "description": "A smooth, side-to-side sway of the entire head.",
        },
    ),
    "dizzy_spin": (
        move_dizzy_spin,
        {
            "roll_amplitude_rad": np.deg2rad(15),
            "pitch_amplitude_rad": np.deg2rad(15),
            "subcycles_per_beat": 0.25,
            **DEFAULT_ANTENNA_PARAMS,
        },
        {
            "default_duration_beats": 4,
            "description": "A circular 'dizzy' head motion combining roll and pitch.",
        },
    ),
    "stumble_and_recover": (
        move_stumble_and_recover,
        {
            "yaw_amplitude_rad": np.deg2rad(25),
            "pitch_amplitude_rad": np.deg2rad(10),
            "y_amplitude_m": 0.015,
            "subcycles_per_beat": 0.25,
            "antenna_move_name": "both",
            "antenna_amplitude_rad": np.deg2rad(50),
        },
        {
            "default_duration_beats": 4,
            "description": "A simulated stumble and recovery with multiple axis movements. Good vibes",
        },
    ),
    "headbanger_combo": (
        move_headbanger_combo,
        {
            "pitch_amplitude_rad": np.deg2rad(30),
            "z_amplitude_m": 0.015,
            "subcycles_per_beat": 1.0,
            "waveform": "sin",
            "antenna_move_name": "both",
            "antenna_amplitude_rad": np.deg2rad(40),
        },
        {
            "default_duration_beats": 4,
            "description": "A strong head nod combined with a vertical bounce.",
        },
    ),
    "interwoven_spirals": (
        move_interwoven_spirals,
        {
            "roll_amp": np.deg2rad(15),
            "pitch_amp": np.deg2rad(20),
            "yaw_amp": np.deg2rad(25),
            "subcycles_per_beat": 0.125,
            **DEFAULT_ANTENNA_PARAMS,
        },
        {
            "default_duration_beats": 8,
            "description": "A complex spiral motion using three axes at different frequencies.",
        },
    ),
    "sharp_side_tilt": (
        move_sharp_side_tilt,
        {
            "roll_amplitude_rad": np.deg2rad(22),
            "subcycles_per_beat": 1.0,
            "waveform": "triangle",
            **DEFAULT_ANTENNA_PARAMS,
        },
        {
            "default_duration_beats": 6,
            "description": "A sharp, quick side-to-side tilt using a triangle waveform.",
        },
    ),
    "side_peekaboo": (
        move_side_peekaboo,
        {
            "z_amp": 0.04,
            "y_amp": 0.03,
            "pitch_amp": np.deg2rad(20),
            "subcycles_per_beat": 0.5,
            "antenna_move_name": "both",
            "antenna_amplitude_rad": np.deg2rad(60),
        },
        {
            "default_duration_beats": 10,
            "description": "A multi-stage peekaboo performance, hiding and peeking to each side.",
        },
    ),
    "yeah_nod": (
        move_yeah_nod,
        {
            "amplitude_rad": np.deg2rad(15),
            "subcycles_per_beat": 1.0,
            "antenna_move_name": "both",
            "antenna_amplitude_rad": np.deg2rad(20),
        },
        {
            "default_duration_beats": 4,
            "description": "An emphatic two-part yeah nod using transient motions.",
        },
    ),
    "uh_huh_tilt": (
        move_uh_huh_tilt,
        {
            "amplitude_rad": np.deg2rad(15),
            "subcycles_per_beat": 0.5,
            **DEFAULT_ANTENNA_PARAMS,
        },
        {
            "default_duration_beats": 4,
            "description": "A combined roll-and-pitch uh-huh gesture of agreement.",
        },
    ),
    "neck_recoil": (
        move_neck_recoil,
        {"amplitude_m": 0.015, "subcycles_per_beat": 0.5, **DEFAULT_ANTENNA_PARAMS},
        {
            "default_duration_beats": 4,
            "description": "A quick, transient backward recoil of the neck.",
        },
    ),
    "chin_lead": (
        move_chin_lead,
        {
            "x_amplitude_m": 0.02,
            "pitch_amplitude_rad": np.deg2rad(15),
            "subcycles_per_beat": 0.5,
            "antenna_move_name": "both",
            **DEFAULT_ANTENNA_PARAMS,
        },
        {
            "default_duration_beats": 4,
            "description": "A forward motion led by the chin, combining translation and pitch.",
        },
    ),
    "groovy_sway_and_roll": (
        move_groovy_sway_and_roll,
        {
            "y_amplitude_m": 0.03,
            "roll_amplitude_rad": np.deg2rad(15),
            "subcycles_per_beat": 0.5,
            **DEFAULT_ANTENNA_PARAMS,
        },
        {
            "default_duration_beats": 4,
            "description": "A side-to-side sway combined with a corresponding roll for a groovy effect.",
        },
    ),
    "chicken_peck": (
        move_chicken_peck,
        {
            "amplitude_m": 0.02,
            "subcycles_per_beat": 1.0,
            "antenna_move_name": "both",
            "antenna_amplitude_rad": np.deg2rad(30),
        },
        {
            "default_duration_beats": 4,
            "description": "A sharp, forward, chicken-like pecking motion.",
        },
    ),
    "side_glance_flick": (
        move_side_glance_flick,
        {
            "yaw_amplitude_rad": np.deg2rad(45),
            "subcycles_per_beat": 0.25,
            **DEFAULT_ANTENNA_PARAMS,
        },
        {
            "default_duration_beats": 4,
            "description": "A quick glance to the side that holds, then returns.",
        },
    ),
    "polyrhythm_combo": (
        move_polyrhythm_combo,
        {
            "sway_amplitude_m": 0.02,
            "nod_amplitude_rad": np.deg2rad(10),
            "antenna_amplitude_rad": np.deg2rad(45),
            "antenna_move_name": "wiggle",
        },
        {
            "default_duration_beats": 6,
            "description": "A 3-beat sway and a 2-beat nod create a polyrhythmic feel.",
        },
    ),
    "grid_snap": (
        move_grid_snap,
        {
            "amplitude_rad": np.deg2rad(20),
            "subcycles_per_beat": 0.25,
            "antenna_move_name": "both",
            "antenna_amplitude_rad": np.deg2rad(10),
        },
        {
            "default_duration_beats": 4,
            "description": "A robotic, grid-snapping motion using square waveforms.",
        },
    ),
    "pendulum_swing": (
        move_pendulum_swing,
        {
            "amplitude_rad": np.deg2rad(25),
            "subcycles_per_beat": 0.25,
            **DEFAULT_ANTENNA_PARAMS,
        },
        {
            "default_duration_beats": 4,
            "description": "A simple, smooth pendulum-like swing using a roll motion.",
        },
    ),
    "jackson_square": (
        move_jackson_square,
        {
            "y_amp_m": 0.035,
            "z_amp_m": 0.025,
            "twitch_amplitude_rad": np.deg2rad(20),
            "z_offset_m": -0.01,
            "subcycles_per_beat": 0.2,
            **DEFAULT_ANTENNA_PARAMS,
        },
        {
            "default_duration_beats": 10,
            "description": "Traces a rectangle via a 5-point path, with sharp twitches on arrival at each checkpoint.",
        },
    ),
}
