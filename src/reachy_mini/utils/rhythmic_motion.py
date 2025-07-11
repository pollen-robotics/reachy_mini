"""A compositional library for creating rhythmic motion in Reachy Mini.

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

Key Parameters
--------------
All move functions are driven by `t_beats` and a set of keyword arguments:
- `t_beats` (float): Elapsed time in **musical beats**. The primary input. t_beats [dimensionless] = time_in_seconds [seconds] * frequency [hertz]
- `amplitude` / `amplitude_rad` / `amplitude_m` (float): The main scale of
  the motion, in radians or meters.
- `subcycles_per_beat` (float): Drives the number of oscillations per beat. 1.0 = one full
  subcycle per beat.
- `phase_offset` (float): A normalized time delay for the motion (0.5 offsets half a period).
- `waveform` (str): The shape of the oscillation (e.g., 'sin', 'square').
- `antenna_move_name` (str): The style of antenna motion ('wiggle', 'both').
- `antenna_amplitude_rad` (float): The scale of the antenna motion.
- `duration_beats` / `repeat_every` (float): Parameters for transient,
  one-shot moves.

Usage Example
-------------

```python
# 1. Get the move function and its default parameters
move_fn = AVAILABLE_DANCE_MOVES['simple_nod']
params = MOVE_SPECIFIC_PARAMS['simple_nod']

# 2. At each time step, call the function with the current beat time
t_beats = 2.5
offsets = move_fn(t_beats, **params)

# 3. Apply the resulting offsets to the robot    "ruff.format": {
        "args": [
            "--line-length",
            "120"
        ]
    }
robot.set_pose(base_pose + offsets.position_offset, ...)
```

"""

from dataclasses import dataclass, replace
from typing import List
import numpy as np
from typing import Callable


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
    duration_in_beats: float = (
        1.0  # The duration, in beats, over which the motion occurs.
    )
    delay_beats: float = 0.0  # An initial delay, in beats, before the motion starts.
    repeat_every: float = (
        0.0  # If > 0, repeat motion at this interval. If 0, it's a one-shot move.
    )


def oscillation_motion(t_beats: float, params: OscillationParams) -> float:
    """Generate an oscillatory motion based on the specified parameters.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        params (OscillationParams): An object containing parameters for the oscillation motion.

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


# TODO: (for later release) Change the behavior so a repeating motions goes backwards to the start to avoid dsiscontinuity.
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
            np.floor((t_beats - params.delay_beats) / params.repeat_every)
            * params.repeat_every
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


# ────────────────────────────── ATOMIC MOVES & HELPERS ──────────────────────────────────
def atomic_x_pos(t_beats: float, params: OscillationParams) -> MoveOffsets:
    """Generate an oscillatory motion offset for the x-axis position.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        params (OscillationParams): Parameters for the oscillation motion.

    Returns:
        MoveOffsets: An object containing the x-axis position offset.

    """
    val = oscillation_motion(t_beats, params)
    return MoveOffsets(np.array([val, 0, 0]), np.zeros(3), np.zeros(2))


def atomic_y_pos(t_beats: float, params: OscillationParams) -> MoveOffsets:
    """Generate an oscillatory motion offset for the y-axis position.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        params (OscillationParams): Parameters for the oscillation motion.

    Returns:
        MoveOffsets: An object containing the y-axis position offset.

    """
    val = oscillation_motion(t_beats, params)
    return MoveOffsets(np.array([0, val, 0]), np.zeros(3), np.zeros(2))


def atomic_z_pos(t_beats: float, params: OscillationParams) -> MoveOffsets:
    """Generate an oscillatory motion offset for the z-axis position.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        params (OscillationParams): Parameters for the oscillation motion.

    Returns:
        MoveOffsets: An object containing the z-axis position offset.

    """
    val = oscillation_motion(t_beats, params)
    return MoveOffsets(np.array([0, 0, val]), np.zeros(3), np.zeros(2))


def atomic_roll(t_beats: float, params: OscillationParams) -> MoveOffsets:
    """Generate an oscillatory motion offset for the roll orientation.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        params (OscillationParams): Parameters for the oscillation motion.

    Returns:
        MoveOffsets: An object containing the roll orientation offset.

    """
    val = oscillation_motion(t_beats, params)
    return MoveOffsets(np.zeros(3), np.array([val, 0, 0]), np.zeros(2))


def atomic_pitch(t_beats: float, params: OscillationParams) -> MoveOffsets:
    """Generate an oscillatory motion offset for the pitch orientation.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        params (OscillationParams): Parameters for the oscillation motion.

    Returns:
        MoveOffsets: An object containing the pitch orientation offset.

    """
    val = oscillation_motion(t_beats, params)
    return MoveOffsets(np.zeros(3), np.array([0, val, 0]), np.zeros(2))


def atomic_yaw(t_beats: float, params: OscillationParams) -> MoveOffsets:
    """Generate an oscillatory motion offset for the yaw orientation.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        params (OscillationParams): Parameters for the oscillation motion.

    Returns:
        MoveOffsets: An object containing the yaw orientation offset.

    """
    val = oscillation_motion(t_beats, params)
    return MoveOffsets(np.zeros(3), np.array([0, 0, val]), np.zeros(2))


def atomic_antenna_wiggle(t_beats: float, params: OscillationParams) -> MoveOffsets:
    """Generate an opposing wiggle motion for the antennas.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        params (OscillationParams): Parameters for the oscillation motion.

    Returns:
        MoveOffsets: An object containing the antenna offsets.

    """
    val = oscillation_motion(t_beats, params)
    return MoveOffsets(np.zeros(3), np.zeros(3), np.array([val, -val]))


def atomic_antenna_both(t_beats: float, params: OscillationParams) -> MoveOffsets:
    """Generate a synchronized motion for both antennas.

    Args:
        t_beats (float): Continuous time in beats at which to evaluate the motion,
            increases by 1 every beat. t_beats [dimensionless] =
            time_in_seconds [seconds] * frequency [hertz].
        params (OscillationParams): Parameters for the oscillation motion.

    Returns:
        MoveOffsets: An object containing the antenna offsets.

    """
    val = oscillation_motion(t_beats, params)
    return MoveOffsets(np.zeros(3), np.zeros(3), np.array([val, val]))


AVAILABLE_ANTENNA_MOVES: dict[
    str, Callable[[float, OscillationParams], MoveOffsets]
] = {"wiggle": atomic_antenna_wiggle, "both": atomic_antenna_both}

# ─────────────────────────── ALL DANCE MOVE FUNCTIONS ─────────────────────────────


def move_simple_nod(
    t_beats,
    amplitude_rad,
    antenna_amplitude_rad,
    subcycles_per_beat,
    antenna_move_name="wiggle",
    phase_offset=0.0,
    waveform="sin",
):
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
    t_beats,
    amplitude_rad,
    antenna_amplitude_rad,
    subcycles_per_beat,
    antenna_move_name="wiggle",
    phase_offset=0.0,
    waveform="sin",
):
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
    t_beats,
    amplitude_m,
    antenna_amplitude_rad,
    subcycles_per_beat,
    antenna_move_name="wiggle",
    phase_offset=0.0,
    waveform="sin",
):
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
    t_beats,
    roll_amplitude_rad,
    pitch_amplitude_rad,
    antenna_amplitude_rad,
    subcycles_per_beat,
    antenna_move_name="wiggle",
    phase_offset=0.0,
    waveform="sin",
):
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
    t_beats,
    yaw_amplitude_rad,
    pitch_amplitude_rad,
    y_amplitude_m,
    antenna_amplitude_rad,
    subcycles_per_beat,
    antenna_move_name="both",
    phase_offset=0.0,
    waveform="sin",
):
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
    t_beats,
    pitch_amplitude_rad,
    z_amplitude_m,
    antenna_amplitude_rad,
    subcycles_per_beat,
    antenna_move_name="both",
    phase_offset=0.0,
    waveform="sin",
):
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
    t_beats,
    roll_amp,
    pitch_amp,
    yaw_amp,
    antenna_amplitude_rad,
    subcycles_per_beat,
    antenna_move_name="wiggle",
    phase_offset=0.0,
    waveform="sin",
):
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
    t_beats,
    roll_amplitude_rad,
    antenna_amplitude_rad,
    subcycles_per_beat,
    antenna_move_name="wiggle",
    phase_offset=0.0,
    waveform="triangle",
):
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
    t_beats,
    z_amp,
    y_amp,
    pitch_amp,
    antenna_amplitude_rad,
    antenna_move_name="both",
    subcycles_per_beat=0.5,
):
    period = 10.0
    t_in_period = t_beats % period
    pos, ori = np.zeros(3), np.zeros(3)

    def ease(t):
        t_clipped = np.clip(t, 0.0, 1.0)
        return t_clipped * t_clipped * (3 - 2 * t_clipped)

    def excited_nod(t):
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
    t_beats,
    amplitude_rad,
    antenna_amplitude_rad,
    subcycles_per_beat,
    antenna_move_name="both",
):
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
    t_beats,
    amplitude_rad,
    antenna_amplitude_rad,
    subcycles_per_beat,
    antenna_move_name="wiggle",
    phase_offset=0.0,
    waveform="sin",
):
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
    t_beats,
    amplitude_m,
    antenna_amplitude_rad,
    subcycles_per_beat,
    antenna_move_name="wiggle",
):
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
    t_beats,
    x_amplitude_m,
    pitch_amplitude_rad,
    antenna_amplitude_rad,
    subcycles_per_beat,
    antenna_move_name="both",
    phase_offset=0.0,
    waveform="sin",
):
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
    t_beats,
    y_amplitude_m,
    roll_amplitude_rad,
    antenna_amplitude_rad,
    subcycles_per_beat,
    antenna_move_name="wiggle",
    phase_offset=0.0,
    waveform="sin",
):
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
    t_beats,
    amplitude_m,
    antenna_amplitude_rad,
    subcycles_per_beat,
    antenna_move_name="both",
):
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
    t_beats,
    yaw_amplitude_rad,
    antenna_amplitude_rad,
    subcycles_per_beat,
    antenna_move_name="wiggle",
):
    period = 1.0 / subcycles_per_beat
    t_in_period = t_beats % period

    def ease(t):
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
    t_beats,
    sway_amplitude_m,
    nod_amplitude_rad,
    antenna_amplitude_rad,
    antenna_move_name="wiggle",
):
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
    t_beats,
    amplitude_rad,
    antenna_amplitude_rad,
    subcycles_per_beat,
    antenna_move_name="both",
    phase_offset=0.0,
):
    base_params = OscillationParams(
        amplitude_rad, subcycles_per_beat, phase_offset, waveform="square"
    )
    pitch_params = replace(base_params, phase_offset=base_params.phase_offset + 0.25)
    antenna_params = OscillationParams(
        antenna_amplitude_rad, subcycles_per_beat, phase_offset
    )
    yaw = atomic_yaw(t_beats, base_params)
    pitch = atomic_pitch(t_beats, pitch_params)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)
    return combine_offsets([yaw, pitch, antennas])


def move_pendulum_swing(
    t_beats,
    amplitude_rad,
    antenna_amplitude_rad,
    subcycles_per_beat,
    antenna_move_name="wiggle",
    phase_offset=0.0,
    waveform="sin",
):
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
    t_beats,
    square_amp_m,
    twitch_amplitude_rad,
    antenna_amplitude_rad,
    subcycles_per_beat,
    antenna_move_name="wiggle",
):
    period = 8.0
    t_in_period = t_beats % period
    pos, ori = np.zeros(3), np.zeros(3)

    def ease(t):
        t_clipped = np.clip(t, 0.0, 1.0)
        return t_clipped * t_clipped * (3 - 2 * t_clipped)

    if t_in_period < 2.0:
        t = (t_in_period - 0.0) / 2.0
        pos[1] = square_amp_m * (1 - 2 * ease(t))
        pos[2] = square_amp_m
    elif t_in_period < 4.0:
        t = (t_in_period - 2.0) / 2.0
        pos[1] = -square_amp_m
        pos[2] = square_amp_m * (1 - 2 * ease(t))
    elif t_in_period < 6.0:
        t = (t_in_period - 4.0) / 2.0
        pos[1] = -square_amp_m * (1 - 2 * ease(t))
        pos[2] = -square_amp_m
    else:
        t = (t_in_period - 6.0) / 2.0
        pos[1] = square_amp_m
        pos[2] = -square_amp_m * (1 - 2 * ease(t))
    twitch_params = TransientParams(
        twitch_amplitude_rad, duration_in_beats=0.2, repeat_every=2.0
    )
    twitch = transient_motion(t_beats, twitch_params)
    twitch_direction = (-1) ** np.floor(t_in_period / 2.0)
    ori[0] = twitch * twitch_direction
    base_move = MoveOffsets(pos, ori, np.zeros(2))
    antenna_params = OscillationParams(antenna_amplitude_rad, subcycles_per_beat)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, antenna_params)
    return combine_offsets([base_move, antennas])


# ────────────────────────── MASTER MOVE DICTIONARIES ──────────────────────────
DEFAULT_ANTENNA_PARAMS = {
    "antenna_move_name": "wiggle",
    "antenna_amplitude_rad": np.deg2rad(45),
}

MOVE_SPECIFIC_PARAMS = {
    # -- Core Rhythms & Validated Classics --
    "simple_nod": {
        "amplitude_rad": np.deg2rad(20),
        "subcycles_per_beat": 1.0,
        **DEFAULT_ANTENNA_PARAMS,
    },
    "head_tilt_roll": {
        "amplitude_rad": np.deg2rad(15),
        "subcycles_per_beat": 0.5,
        **DEFAULT_ANTENNA_PARAMS,
    },
    "side_to_side_sway": {
        "amplitude_m": 0.04,
        "subcycles_per_beat": 0.5,
        **DEFAULT_ANTENNA_PARAMS,
    },
    "dizzy_spin": {
        "roll_amplitude_rad": np.deg2rad(15),
        "pitch_amplitude_rad": np.deg2rad(15),
        "subcycles_per_beat": 0.25,
        **DEFAULT_ANTENNA_PARAMS,
    },
    "stumble_and_recover": {
        "yaw_amplitude_rad": np.deg2rad(25),
        "pitch_amplitude_rad": np.deg2rad(10),
        "y_amplitude_m": 0.015,
        "subcycles_per_beat": 0.25,
        "antenna_move_name": "both",
        "antenna_amplitude_rad": np.deg2rad(50),
    },
    "headbanger_combo": {
        "pitch_amplitude_rad": np.deg2rad(30),
        "z_amplitude_m": 0.015,
        "subcycles_per_beat": 1.0,
        "waveform": "sin",
        "antenna_move_name": "both",
        "antenna_amplitude_rad": np.deg2rad(40),
    },
    "interwoven_spirals": {
        "roll_amp": np.deg2rad(15),
        "pitch_amp": np.deg2rad(20),
        "yaw_amp": np.deg2rad(25),
        "subcycles_per_beat": 0.125,
        **DEFAULT_ANTENNA_PARAMS,
    },
    "sharp_side_tilt": {
        "roll_amplitude_rad": np.deg2rad(22),
        "subcycles_per_beat": 1.0,
        "waveform": "triangle",
        **DEFAULT_ANTENNA_PARAMS,
    },
    "side_peekaboo": {
        "z_amp": 0.04,
        "y_amp": 0.03,
        "pitch_amp": np.deg2rad(20),
        "subcycles_per_beat": 0.5,
        "antenna_move_name": "both",
        "antenna_amplitude_rad": np.deg2rad(60),
    },
    # -- Groove & Funk --
    "yeah_nod": {
        "amplitude_rad": np.deg2rad(15),
        "subcycles_per_beat": 1.0,
        "antenna_move_name": "both",
        "antenna_amplitude_rad": np.deg2rad(20),
    },
    "uh_huh_tilt": {
        "amplitude_rad": np.deg2rad(15),
        "subcycles_per_beat": 0.5,
        **DEFAULT_ANTENNA_PARAMS,
    },
    "neck_recoil": {
        "amplitude_m": 0.015,
        "subcycles_per_beat": 0.5,
        **DEFAULT_ANTENNA_PARAMS,
    },
    "chin_lead": {
        "x_amplitude_m": 0.02,
        "pitch_amplitude_rad": np.deg2rad(15),
        "subcycles_per_beat": 0.5,
        "antenna_move_name": "both",
        **DEFAULT_ANTENNA_PARAMS,
    },
    "groovy_sway_and_roll": {
        "y_amplitude_m": 0.03,
        "roll_amplitude_rad": np.deg2rad(15),
        "subcycles_per_beat": 0.5,
        **DEFAULT_ANTENNA_PARAMS,
    },
    "chicken_peck": {
        "amplitude_m": 0.02,
        "subcycles_per_beat": 1.0,
        "antenna_move_name": "both",
        "antenna_amplitude_rad": np.deg2rad(30),
    },
    # -- Sassy & Expressive --
    "side_glance_flick": {
        "yaw_amplitude_rad": np.deg2rad(45),
        "subcycles_per_beat": 0.25,
        **DEFAULT_ANTENNA_PARAMS,
    },
    "polyrhythm_combo": {
        "sway_amplitude_m": 0.02,
        "nod_amplitude_rad": np.deg2rad(10),
        "antenna_amplitude_rad": np.deg2rad(45),
        "antenna_move_name": "wiggle",
    },
    # -- Robotic & Glitch --
    "grid_snap": {
        "amplitude_rad": np.deg2rad(20),
        "subcycles_per_beat": 0.25,
        "antenna_move_name": "both",
        "antenna_amplitude_rad": np.deg2rad(10),
    },
    "pendulum_swing": {
        "amplitude_rad": np.deg2rad(25),
        "subcycles_per_beat": 0.25,
        **DEFAULT_ANTENNA_PARAMS,
    },
    "jackson_square": {
        "square_amp_m": 0.035,
        "twitch_amplitude_rad": np.deg2rad(30),
        "subcycles_per_beat": 0.125,
        **DEFAULT_ANTENNA_PARAMS,
    },
}

AVAILABLE_DANCE_MOVES: dict[str, Callable] = {
    # -- Core Rhythms & Validated Classics --
    "simple_nod": move_simple_nod,
    "head_tilt_roll": move_head_tilt_roll,
    "side_to_side_sway": move_side_to_side_sway,
    "dizzy_spin": move_dizzy_spin,
    "stumble_and_recover": move_stumble_and_recover,
    "headbanger_combo": move_headbanger_combo,
    "interwoven_spirals": move_interwoven_spirals,
    "sharp_side_tilt": move_sharp_side_tilt,
    "side_peekaboo": move_side_peekaboo,
    # -- Groove & Funk --
    "yeah_nod": move_yeah_nod,
    "uh_huh_tilt": move_uh_huh_tilt,
    "neck_recoil": move_neck_recoil,
    "chin_lead": move_chin_lead,
    "groovy_sway_and_roll": move_groovy_sway_and_roll,
    "chicken_peck": move_chicken_peck,
    # -- Sassy & Expressive --
    "side_glance_flick": move_side_glance_flick,
    "polyrhythm_combo": move_polyrhythm_combo,
    # -- Robotic & Glitch --
    "grid_snap": move_grid_snap,
    "pendulum_swing": move_pendulum_swing,
    "jackson_square": move_jackson_square,
}


def _test_transient_motion():
    """Run a few examples of transient_motion and print the output."""
    print("=" * 50)
    print(" DEMONSTRATION OF transient_motion() ".center(50, "="))
    print("=" * 50)
    print("\n--- Case 1: Simple One-Shot (duration=2.0) ---")
    print(
        "The motion should start at t=0.0, hit its peak at t=2.0, and then stay at 0."
    )
    params1 = TransientParams(amplitude=10.0, duration_in_beats=2.0)
    for t in np.arange(0, 4.25, 0.25):
        print(f"t={t:4.2f} -> value={transient_motion(t, params1):6.3f}")
    print("\n--- Case 2: One-Shot with Delay (duration=2.0, delay=1.0) ---")
    print(
        "The motion should be 0 until t=1.0, then start, finish at t=3.0, and then stay 0."
    )
    params2 = TransientParams(amplitude=10.0, duration_in_beats=2.0, delay_beats=1.0)
    for t in np.arange(0, 5.25, 0.25):
        print(f"t={t:4.2f} -> value={transient_motion(t, params2):6.3f}")
    print("\n--- Case 3: Repeating Motion (duration=1.0, repeat_every=4.0) ---")
    print(
        "A short 1-beat motion will occur every 4 beats. Look for motion during [0,1), [4,5), and [8,9)."
    )
    params3 = TransientParams(amplitude=5.0, duration_in_beats=1.0, repeat_every=4.0)
    for t in np.arange(0, 10.25, 0.25):
        print(f"t={t:4.2f} -> value={transient_motion(t, params3):6.3f}")
    print("\n" + "=" * 50)


if __name__ == "__main__":
    _test_transient_motion()
