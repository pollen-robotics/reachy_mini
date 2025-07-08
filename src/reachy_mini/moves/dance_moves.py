"""Dance Motion.

Key Parameters
--------------
All move functions are driven by `t_beats` and a set of keyword arguments:
- `t_beats` (float): Elapsed time in **musical beats**. The primary input.
- `amplitude` / `amplitude_rad` / `amplitude_m` (float): The main scale of
  the motion, in radians or meters.
- `cycles_per_beat` (float): The speed of the motion. 1.0 = one full
  cycle per beat.
- `phase_offset` (float): A time delay for the motion, in cycles (0.0-1.0).
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

# 3. Apply the resulting offsets to the robot
robot.set_pose(base_pose + offsets.position_offset, ...)
```
"""

from typing import Callable

import numpy as np

from reachy_mini.moves.utils import (
    MoveOffsets,
    OscillationParams,
    combine_offsets,
    oscillation_motion,
    transient_motion,
)


# ────────────────────────────── ATOMIC MOVES & HELPERS ──────────────────────────────────
def atomic_x_pos(t_beats: float, params: OscillationParams) -> "MoveOffsets":
    """Generate an oscillatory motion offset for the x-axis position.

    Args:
        t_beats (float): Time in beats at which to evaluate the motion. Beware that this is not the same as time in seconds but in beats (dimensionless).
        params (OscillationParams): Parameters for the oscillation motion.

    Returns:
        MoveOffsets: An object containing the x-axis position offset and zero offsets for other axes and orientations.

    """
    return MoveOffsets(
        np.array(
            [
                oscillation_motion(t_beats, params),
                0,
                0,
            ]
        ),
        np.zeros(3),
        np.zeros(2),
    )


def atomic_y_pos(t_beats: float, params: OscillationParams) -> "MoveOffsets":
    """Generate an oscillatory motion offset for the y-axis position.

    Args:
        t_beats (float): Time in beats at which to evaluate the motion. Beware that this is not the same as time in seconds but in beats (dimensionless).
        params (OscillationParams): Parameters for the oscillation motion.

    Returns:
        MoveOffsets: An object containing the y-axis position offset and zero offsets for other axes and orientations.

    """
    return MoveOffsets(
        np.array([0, oscillation_motion(t_beats, params), 0]),
        np.zeros(3),
        np.zeros(2),
    )


def atomic_z_pos(t_beats: float, params: OscillationParams) -> "MoveOffsets":
    """Generate an oscillatory motion offset for the z-axis position.

    Args:
        t_beats (float): Time in beats at which to evaluate the motion.
        params (OscillationParams, optional): Parameters for the oscillation motion.

    Returns:
        MoveOffsets: An object containing the z-axis position offset and zero offsets for other axes and orientations.

    """
    return MoveOffsets(
        np.array([0, 0, oscillation_motion(t_beats, params)]),
        np.zeros(3),
        np.zeros(2),
    )


def atomic_roll(t_beats: float, params: OscillationParams) -> "MoveOffsets":
    """Generate an oscillatory motion offset for the roll orientation.

    Args:
        t_beats (float): Time in beats at which to evaluate the motion.
        params (OscillationParams, optional): Parameters for the oscillation motion.

    Returns:
        MoveOffsets: An object containing the roll orientation offset and zero offsets for other axes and orientations.

    """
    return MoveOffsets(
        np.zeros(3),
        np.array([oscillation_motion(t_beats, params), 0, 0]),
        np.zeros(2),
    )


def atomic_pitch(t_beats, **kwargs):
    return MoveOffsets(
        np.zeros(3),
        np.array([0, oscillation_motion(t_beats, **kwargs), 0]),
        np.zeros(2),
    )


def atomic_yaw(t_beats, **kwargs):
    return MoveOffsets(
        np.zeros(3),
        np.array([0, 0, oscillation_motion(t_beats, **kwargs)]),
        np.zeros(2),
    )


def atomic_antenna_wiggle(t_beats, **kwargs):
    return MoveOffsets(
        np.zeros(3),
        np.zeros(3),
        np.array(
            [
                oscillation_motion(t_beats, **kwargs),
                -oscillation_motion(t_beats, **kwargs),
            ]
        ),
    )


def atomic_antenna_both(t_beats, **kwargs):
    return MoveOffsets(
        np.zeros(3), np.zeros(3), np.array([oscillation_motion(t_beats, **kwargs)] * 2)
    )


AVAILABLE_ANTENNA_MOVES = {"wiggle": atomic_antenna_wiggle, "both": atomic_antenna_both}
VALID_OSCILLATION_KEYS = {"cycles_per_beat", "phase_offset", "waveform"}


def get_base_move_kwargs(main_kwargs):
    return {
        key: main_kwargs[key] for key in VALID_OSCILLATION_KEYS if key in main_kwargs
    }


def get_antenna_kwargs(main_kwargs):
    antenna_params = get_base_move_kwargs(main_kwargs)
    if "antenna_amplitude_rad" in main_kwargs:
        antenna_params["amplitude"] = main_kwargs["antenna_amplitude_rad"]
    return antenna_params


# ─────────────────────────── ALL DANCE MOVE FUNCTIONS ─────────────────────────────


def move_simple_nod(t_beats, amplitude_rad, antenna_move_name="wiggle", **kwargs):
    base = atomic_pitch(
        t_beats, amplitude=amplitude_rad, **get_base_move_kwargs(kwargs)
    )
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](
        t_beats, **get_antenna_kwargs(kwargs)
    )
    return combine_offsets(base, antennas)


def move_head_tilt_roll(t_beats, amplitude_rad, antenna_move_name="wiggle", **kwargs):
    base = atomic_roll(t_beats, amplitude=amplitude_rad, **get_base_move_kwargs(kwargs))
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](
        t_beats, **get_antenna_kwargs(kwargs)
    )
    return combine_offsets(base, antennas)


def move_side_to_side_sway(t_beats, amplitude_m, antenna_move_name="wiggle", **kwargs):
    base = atomic_y_pos(t_beats, amplitude=amplitude_m, **get_base_move_kwargs(kwargs))
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](
        t_beats, **get_antenna_kwargs(kwargs)
    )
    return combine_offsets(base, antennas)


def move_dizzy_spin(
    t_beats,
    roll_amplitude_rad,
    pitch_amplitude_rad,
    antenna_move_name="wiggle",
    **kwargs,
):
    base_kwargs = get_base_move_kwargs(kwargs)
    roll = atomic_roll(t_beats, amplitude=roll_amplitude_rad, **base_kwargs)
    pitch_kwargs = base_kwargs.copy()
    pitch_kwargs["phase_offset"] = pitch_kwargs.get("phase_offset", 0) + 0.25
    pitch = atomic_pitch(t_beats, amplitude=pitch_amplitude_rad, **pitch_kwargs)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](
        t_beats, **get_antenna_kwargs(kwargs)
    )
    return combine_offsets(roll, pitch, antennas)


def move_stumble_and_recover(
    t_beats,
    yaw_amplitude_rad,
    pitch_amplitude_rad,
    y_amplitude_m,
    antenna_move_name="both",
    **kwargs,
):
    base_kwargs = get_base_move_kwargs(kwargs)
    yaw = atomic_yaw(t_beats, amplitude=yaw_amplitude_rad, **base_kwargs)
    pitch_kwargs = base_kwargs.copy()
    pitch_kwargs["cycles_per_beat"] = pitch_kwargs.get("cycles_per_beat", 1.0) * 2
    pitch = atomic_pitch(t_beats, amplitude=pitch_amplitude_rad, **pitch_kwargs)
    sway_kwargs = base_kwargs.copy()
    sway_kwargs["phase_offset"] = sway_kwargs.get("phase_offset", 0) + 0.5
    stabilizer_sway = atomic_y_pos(t_beats, amplitude=y_amplitude_m, **sway_kwargs)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](
        t_beats, **get_antenna_kwargs(kwargs)
    )
    return combine_offsets(yaw, pitch, stabilizer_sway, antennas)


def move_headbanger_combo(
    t_beats, pitch_amplitude_rad, z_amplitude_m, antenna_move_name="both", **kwargs
):
    base_kwargs = get_base_move_kwargs(kwargs)
    nod = atomic_pitch(t_beats, amplitude=pitch_amplitude_rad, **base_kwargs)
    bounce_kwargs = base_kwargs.copy()
    bounce_kwargs["phase_offset"] = bounce_kwargs.get("phase_offset", 0) + 0.1
    bounce = atomic_z_pos(t_beats, amplitude=z_amplitude_m, **bounce_kwargs)
    antenna_kwargs = get_antenna_kwargs(kwargs)
    antenna_kwargs["amplitude"] = pitch_amplitude_rad * 2.0
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, **antenna_kwargs)
    return combine_offsets(nod, bounce, antennas)


def move_interwoven_spirals(
    t_beats, roll_amp, pitch_amp, yaw_amp, antenna_move_name="wiggle", **kwargs
):
    base_kwargs = get_base_move_kwargs(kwargs)
    roll_kwargs = base_kwargs.copy()
    roll_kwargs["cycles_per_beat"] = 0.125
    pitch_kwargs = base_kwargs.copy()
    pitch_kwargs.update(
        {
            "cycles_per_beat": 0.25,
            "phase_offset": base_kwargs.get("phase_offset", 0) + 0.25,
        }
    )
    yaw_kwargs = base_kwargs.copy()
    yaw_kwargs.update(
        {
            "cycles_per_beat": 0.5,
            "phase_offset": base_kwargs.get("phase_offset", 0) + 0.5,
        }
    )
    roll = atomic_roll(t_beats, amplitude=roll_amp, **roll_kwargs)
    pitch = atomic_pitch(t_beats, amplitude=pitch_amp, **pitch_kwargs)
    yaw = atomic_yaw(t_beats, amplitude=yaw_amp, **yaw_kwargs)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](
        t_beats, **get_antenna_kwargs(kwargs)
    )
    return combine_offsets(roll, pitch, yaw, antennas)


def move_sharp_side_tilt(
    t_beats, roll_amplitude_rad, antenna_move_name="wiggle", **kwargs
):
    base = atomic_roll(
        t_beats, amplitude=roll_amplitude_rad, **get_base_move_kwargs(kwargs)
    )
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](
        t_beats, **get_antenna_kwargs(kwargs)
    )
    return combine_offsets(base, antennas)


def move_side_peekaboo(
    t_beats, z_amp, y_amp, pitch_amp, antenna_move_name="both", **kwargs
):
    period = 8.0
    t_in_period = t_beats % period
    pos, ori = np.zeros(3), np.zeros(3)

    def ease(t):
        return t * t * (3 - 2 * t)

    def excited_nod(t):
        return pitch_amp * np.sin(t * np.pi)

    if t_in_period < 2.0:
        t = t_in_period / 2.0
        pos[2], pos[1], ori[1] = -z_amp * (1 - ease(t)), y_amp * ease(t), excited_nod(t)
    elif t_in_period < 4.0:
        t = (t_in_period - 2.0) / 2.0
        pos[2], pos[1] = -z_amp * ease(t), y_amp * (1 - ease(t))
    elif t_in_period < 6.0:
        t = (t_in_period - 4.0) / 2.0
        pos[2], pos[1], ori[1] = (
            -z_amp * (1 - ease(t)),
            -y_amp * ease(t),
            -excited_nod(t),
        )
    else:
        t = (t_in_period - 6.0) / 2.0
        pos[2], pos[1] = -z_amp * ease(t), -y_amp * (1 - ease(t))
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](
        t_beats, **get_antenna_kwargs(kwargs)
    )
    return combine_offsets(MoveOffsets(pos, ori, np.zeros(2)), antennas)


def move_yeah_nod(t_beats, amplitude_rad, antenna_move_name="both", **kwargs):  # FIXED
    base_kwargs = get_base_move_kwargs(kwargs)
    repeat_every = 1.0 / base_kwargs.get("cycles_per_beat", 1.0)
    nod1 = transient_motion(
        t_beats,
        amplitude_rad,
        duration_beats=repeat_every * 0.4,
        repeat_every=repeat_every,
    )
    nod2 = transient_motion(
        t_beats,
        amplitude_rad * 0.7,
        duration_beats=repeat_every * 0.3,
        delay_beats=repeat_every * 0.5,
        repeat_every=repeat_every,
    )
    base = MoveOffsets(np.zeros(3), np.array([0, nod1 + nod2, 0]), np.zeros(2))
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](
        t_beats, **get_antenna_kwargs(kwargs)
    )
    return combine_offsets(base, antennas)


def move_uh_huh_tilt(t_beats, amplitude_rad, antenna_move_name="wiggle", **kwargs):
    base_kwargs = get_base_move_kwargs(kwargs)
    roll = atomic_roll(t_beats, amplitude=amplitude_rad, **base_kwargs)
    pitch = atomic_pitch(t_beats, amplitude=amplitude_rad, **base_kwargs)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](
        t_beats, **get_antenna_kwargs(kwargs)
    )
    return combine_offsets(roll, pitch, antennas)


def move_neck_recoil(
    t_beats, amplitude_m, antenna_move_name="wiggle", **kwargs
):  # FIXED
    base_kwargs = get_base_move_kwargs(kwargs)
    repeat_every = 1.0 / base_kwargs.get("cycles_per_beat", 0.5)
    recoil = transient_motion(
        t_beats,
        -amplitude_m,
        duration_beats=repeat_every * 0.3,
        repeat_every=repeat_every,
    )
    base = MoveOffsets(np.array([recoil, 0, 0]), np.zeros(3), np.zeros(2))
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](
        t_beats, **get_antenna_kwargs(kwargs)
    )
    return combine_offsets(base, antennas)


def move_chin_lead(
    t_beats, x_amplitude_m, pitch_amplitude_rad, antenna_move_name="both", **kwargs
):
    base_kwargs = get_base_move_kwargs(kwargs)
    x_move = atomic_x_pos(t_beats, amplitude=x_amplitude_m, **base_kwargs)
    pitch_kwargs = base_kwargs.copy()
    pitch_kwargs["phase_offset"] = pitch_kwargs.get("phase_offset", 0) - 0.25
    pitch_move = atomic_pitch(t_beats, amplitude=pitch_amplitude_rad, **pitch_kwargs)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](
        t_beats, **get_antenna_kwargs(kwargs)
    )
    return combine_offsets(x_move, pitch_move, antennas)


def move_groovy_sway_and_roll(
    t_beats, y_amplitude_m, roll_amplitude_rad, antenna_move_name="wiggle", **kwargs
):
    base_kwargs = get_base_move_kwargs(kwargs)
    sway = atomic_y_pos(t_beats, amplitude=y_amplitude_m, **base_kwargs)
    roll_kwargs = base_kwargs.copy()
    roll_kwargs["phase_offset"] = roll_kwargs.get("phase_offset", 0) + 0.25
    roll = atomic_roll(t_beats, amplitude=roll_amplitude_rad, **roll_kwargs)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](
        t_beats, **get_antenna_kwargs(kwargs)
    )
    return combine_offsets(sway, roll, antennas)


def move_chicken_peck(t_beats, amplitude_m, antenna_move_name="both", **kwargs):
    base_kwargs = get_base_move_kwargs(kwargs)
    repeat_every = 1.0 / base_kwargs.get("cycles_per_beat", 1.0)
    x_offset = transient_motion(
        t_beats,
        amplitude_m,
        duration_beats=repeat_every * 0.8,
        repeat_every=repeat_every,
    )
    pitch_offset = transient_motion(
        t_beats,
        amplitude_m * 5,
        duration_beats=repeat_every * 0.8,
        repeat_every=repeat_every,
    )
    antenna_kwargs = get_antenna_kwargs(kwargs)
    antenna_kwargs["amplitude"] = antenna_kwargs.get("amplitude", 1.0) * 0.5
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, **antenna_kwargs)
    return combine_offsets(
        MoveOffsets(
            np.array([x_offset, 0, 0]), np.array([0, pitch_offset, 0]), np.zeros(2)
        ),
        antennas,
    )


def move_side_glance_flick(
    t_beats, yaw_amplitude_rad, antenna_move_name="wiggle", **kwargs
):  # FIXED
    period = 4.0
    t_in_period = t_beats % (1.0 / kwargs.get("cycles_per_beat", 0.25))

    def ease(t):
        return t * t * (3 - 2 * t)

    yaw_offset = 0
    if t_in_period < 0.5:
        yaw_offset = yaw_amplitude_rad * ease(t_in_period / 0.5)
    elif t_in_period < 1.5:
        yaw_offset = yaw_amplitude_rad
    else:
        yaw_offset = yaw_amplitude_rad * (
            1.0 - ease((t_in_period - 1.5) / (period - 1.5))
        )
    base = MoveOffsets(np.zeros(3), np.array([0, 0, yaw_offset]), np.zeros(2))
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](
        t_beats, **get_antenna_kwargs(kwargs)
    )
    return combine_offsets(base, antennas)


def move_polyrhythm_combo(
    t_beats, sway_amplitude_m, nod_amplitude_rad, antenna_move_name="wiggle", **kwargs
):
    sway = atomic_y_pos(t_beats, amplitude=sway_amplitude_m, cycles_per_beat=1 / 3)
    nod = atomic_pitch(t_beats, amplitude=nod_amplitude_rad, cycles_per_beat=1 / 2)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](
        t_beats, **get_antenna_kwargs(kwargs)
    )
    return combine_offsets(sway, nod, antennas)


def move_grid_snap(t_beats, amplitude_rad, antenna_move_name="both", **kwargs):
    base_kwargs = get_base_move_kwargs(kwargs)
    base_kwargs["waveform"] = "square"
    yaw = atomic_yaw(t_beats, amplitude=amplitude_rad, **base_kwargs)
    pitch_kwargs = base_kwargs.copy()
    pitch_kwargs["phase_offset"] = pitch_kwargs.get("phase_offset", 0) + 0.25
    pitch = atomic_pitch(t_beats, amplitude=amplitude_rad, **pitch_kwargs)
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](
        t_beats, **get_antenna_kwargs(kwargs)
    )
    return combine_offsets(yaw, pitch, antennas)


def move_pendulum_swing(t_beats, amplitude_rad, antenna_move_name="wiggle", **kwargs):
    base = atomic_roll(t_beats, amplitude=amplitude_rad, **get_base_move_kwargs(kwargs))
    antennas = AVAILABLE_ANTENNA_MOVES[antenna_move_name](
        t_beats, **get_antenna_kwargs(kwargs)
    )
    return combine_offsets(base, antennas)


# ────────────────────────── MASTER MOVE DICTIONARIES ──────────────────────────
DEFAULT_ANTENNA_PARAMS = {
    "antenna_move_name": "wiggle",
    "antenna_amplitude_rad": np.deg2rad(45),
}

MOVE_SPECIFIC_PARAMS = {
    # -- Core Rhythms & Validated Classics --
    "simple_nod": {
        "amplitude_rad": np.deg2rad(20),
        "cycles_per_beat": 1.0,
        **DEFAULT_ANTENNA_PARAMS,
    },
    "head_tilt_roll": {
        "amplitude_rad": np.deg2rad(15),
        "cycles_per_beat": 0.5,
        **DEFAULT_ANTENNA_PARAMS,
    },
    "side_to_side_sway": {
        "amplitude_m": 0.04,
        "cycles_per_beat": 0.5,
        **DEFAULT_ANTENNA_PARAMS,
    },
    "dizzy_spin": {
        "roll_amplitude_rad": np.deg2rad(15),
        "pitch_amplitude_rad": np.deg2rad(15),
        "cycles_per_beat": 0.25,
        **DEFAULT_ANTENNA_PARAMS,
    },  # TUNED
    "stumble_and_recover": {
        "yaw_amplitude_rad": np.deg2rad(25),
        "pitch_amplitude_rad": np.deg2rad(10),
        "y_amplitude_m": 0.015,
        "cycles_per_beat": 0.25,
        "antenna_move_name": "both",
        "antenna_amplitude_rad": np.deg2rad(50),
    },
    "headbanger_combo": {
        "pitch_amplitude_rad": np.deg2rad(30),
        "z_amplitude_m": 0.015,
        "cycles_per_beat": 1.0,
        "waveform": "sin",
        "antenna_move_name": "both",
        "antenna_amplitude_rad": np.deg2rad(40),
    },
    "interwoven_spirals": {
        "roll_amp": np.deg2rad(15),
        "pitch_amp": np.deg2rad(20),
        "yaw_amp": np.deg2rad(25),
        "cycles_per_beat": 0.125,
        **DEFAULT_ANTENNA_PARAMS,
    },  # TUNED
    "sharp_side_tilt": {
        "roll_amplitude_rad": np.deg2rad(22),
        "cycles_per_beat": 1.0,
        "waveform": "triangle",
        **DEFAULT_ANTENNA_PARAMS,
    },
    "side_peekaboo": {
        "z_amp": 0.04,
        "y_amp": 0.03,
        "pitch_amp": np.deg2rad(20),
        "cycles_per_beat": 0.5,
        "antenna_move_name": "both",
        "antenna_amplitude_rad": np.deg2rad(60),
    },
    # -- Groove & Funk --
    "yeah_nod": {
        "amplitude_rad": np.deg2rad(15),
        "cycles_per_beat": 1.0,
        "antenna_move_name": "both",
        "antenna_amplitude_rad": np.deg2rad(20),
    },
    "uh_huh_tilt": {
        "amplitude_rad": np.deg2rad(15),
        "cycles_per_beat": 0.5,
        **DEFAULT_ANTENNA_PARAMS,
    },
    "neck_recoil": {
        "amplitude_m": 0.015,
        "cycles_per_beat": 0.5,
        **DEFAULT_ANTENNA_PARAMS,
    },
    "chin_lead": {
        "x_amplitude_m": 0.02,
        "pitch_amplitude_rad": np.deg2rad(15),
        "cycles_per_beat": 0.5,
        **DEFAULT_ANTENNA_PARAMS,
    },
    "groovy_sway_and_roll": {
        "y_amplitude_m": 0.03,
        "roll_amplitude_rad": np.deg2rad(15),
        "cycles_per_beat": 0.5,
        **DEFAULT_ANTENNA_PARAMS,
    },
    "chicken_peck": {
        "amplitude_m": 0.02,
        "cycles_per_beat": 1.0,
        "antenna_move_name": "both",
        "antenna_amplitude_rad": np.deg2rad(30),
    },
    # -- Sassy & Expressive --
    "side_glance_flick": {
        "yaw_amplitude_rad": np.deg2rad(45),
        "cycles_per_beat": 0.25,
        **DEFAULT_ANTENNA_PARAMS,
    },
    "polyrhythm_combo": {
        "sway_amplitude_m": 0.02,
        "nod_amplitude_rad": np.deg2rad(10),
        "cycles_per_beat": 1.0,
        **DEFAULT_ANTENNA_PARAMS,
    },
    # -- Robotic & Glitch --
    "grid_snap": {
        "amplitude_rad": np.deg2rad(20),
        "cycles_per_beat": 0.25,
        "antenna_move_name": "both",
        "antenna_amplitude_rad": np.deg2rad(10),
    },
    "pendulum_swing": {
        "amplitude_rad": np.deg2rad(25),
        "cycles_per_beat": 0.25,
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
}
