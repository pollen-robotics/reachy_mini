"""
Dance Motion Library
---------------------
This module provides beat-synchronized motion primitives for robotic or animated characters.
Each motion function takes as input a normalized time parameter `t_beats`, representing elapsed musical time in **beats**, not seconds.

Key Concepts:
- `t_beats` (float): Elapsed time in **musical beats** (dimensionless). 1.0 = one beat.
  Example: If BPM = 120 and elapsed time = 1.5s, then `t_beats = (120 / 60) * 1.5 = 3.0`
- `cycles_per_beat` (float): Number of full motion cycles per beat. For example, 1.0 means one full cycle per beat.
- `phase_offset` (float): Phase shift in **cycles** (0.0–1.0 wraps once). 0.5 means a half-cycle delay.
- `waveform` (str): Type of periodic function to use ('sin', 'cos', 'triangle', 'square', etc).

Returns:
- All motion functions return a `MoveOffsets` dataclass with:
    - `position_offset` (np.ndarray, shape (3,)): X, Y, Z offsets in meters
    - `orientation_offset` (np.ndarray, shape (3,)): Roll, Pitch, Yaw offsets in radians
    - `antennas_offset` (np.ndarray, shape (2,)): 2 antenna angles in radians
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass
class MoveOffsets:
    """A structured container for position, orientation, and antenna offsets."""
    position_offset: np.ndarray
    orientation_offset: np.ndarray
    antennas_offset: np.ndarray

# --- Oscillation Primitives ---
def oscillation_motion(t_beats: float, amplitude: float, cycles_per_beat: float = 1.0, phase_offset: float = 0.0, waveform: str = 'sin') -> float:
    """Generates a periodic value based on a selected waveform."""
    phase = 2 * np.pi * (cycles_per_beat * t_beats + phase_offset)
    if waveform == 'sin':
        return amplitude * np.sin(phase)
    elif waveform == 'cos':
        return amplitude * np.cos(phase)
    elif waveform == 'square':
        return amplitude * np.sign(np.sin(phase))
    elif waveform == 'triangle':
        # Linear approximation of a triangle wave
        return amplitude * (2 / np.pi) * np.arcsin(np.sin(phase))
    elif waveform == 'sawtooth':
        # This creates a rising sawtooth wave
        return amplitude * (2 * ( (phase / (2 * np.pi)) % 1) - 1)
    else:
        raise ValueError(f"Unsupported waveform type: {waveform}")

# --- Individual Move Parameters ---
MOVE_SPECIFIC_PARAMS = {
    "simple_nod": {"amplitude_rad": np.deg2rad(20), "cycles_per_beat": 1.0, "antenna_move_name": "wiggle"},
    "head_bob_z": {"amplitude_m": 0.03, "cycles_per_beat": 1.0, "antenna_move_name": "both_forward_back"},
    "side_to_side_sway": {"amplitude_m": 0.04, "cycles_per_beat": 0.5, "antenna_move_name": "wiggle"},
    "head_tilt_roll": {"amplitude_rad": np.deg2rad(25), "cycles_per_beat": 0.5, "antenna_move_name": "wiggle"},
    "look_around_yaw": {"amplitude_rad": np.deg2rad(30), "cycles_per_beat": 0.25, "antenna_move_name": "none"},
    "circular_head_roll": {"pitch_amplitude_rad": np.deg2rad(15), "roll_amplitude_rad": np.deg2rad(15), "cycles_per_beat": 0.25, "antenna_move_name": "both_forward_back"},
    "robot_z_bounce": {"amplitude_m": 0.03, "cycles_per_beat": 0.5, "antenna_move_name": "wiggle"},
    "floating_x_glide": {"amplitude_m": 0.02, "cycles_per_beat": 1.0, "antenna_move_name": "wiggle"},
    "side_to_side_head_bob_what_is_love": {"roll_amplitude_rad": np.deg2rad(20), "cycles_per_beat": 1.0, "antenna_move_name": "wiggle"},
    "figure_eight": {"yaw_amplitude_rad": np.deg2rad(20), "pitch_amplitude_rad": np.deg2rad(15), "cycles_per_beat": 0.25, "antenna_move_name": "wiggle"},
}

def get_move_parameters(move_name: str) -> dict:
    """Retrieves the default parameters for a given move name."""
    return MOVE_SPECIFIC_PARAMS.get(move_name, {})

# --- Antenna-only Move Functions ---
# (This section is unchanged)
def move_antenna_wiggle(t_beats: float, amplitude_rad: float = 0.7, cycles_per_beat: float = 0.5) -> np.ndarray:
    val = oscillation_motion(t_beats, amplitude_rad, cycles_per_beat)
    return np.array([val, -val])

def move_antenna_both_forward_back(t_beats: float, amplitude_rad: float = 0.6, cycles_per_beat: float = 1.0) -> np.ndarray:
    val = oscillation_motion(t_beats, amplitude_rad, cycles_per_beat)
    return np.array([val, val])

def move_antenna_none(t_beats: float) -> np.ndarray:
    return np.zeros(2)

AVAILABLE_ANTENNA_MOVES: dict[str, Callable] = {
    "wiggle": move_antenna_wiggle,
    "both_forward_back": move_antenna_both_forward_back,
    "none": move_antenna_none,
}

# --- Core Dance Moves ---
# (Simple moves are unchanged)
def move_simple_nod(t_beats: float, amplitude_rad: float = np.deg2rad(15), cycles_per_beat: float = 1.0, phase_offset: float = 0.0, waveform: str = 'sin', antenna_move_name: str = 'wiggle') -> MoveOffsets:
    pitch = oscillation_motion(t_beats, amplitude_rad, cycles_per_beat, phase_offset, waveform)
    antennas = AVAILABLE_ANTENNA_MOVES.get(antenna_move_name, move_antenna_none)(t_beats)
    return MoveOffsets(np.zeros(3), np.array([0.0, pitch, 0.0]), antennas)

def move_head_bob_z(t_beats: float, amplitude_m: float = 0.02, cycles_per_beat: float = 1.0, phase_offset: float = 0.0, waveform: str = 'sin', antenna_move_name: str = 'wiggle') -> MoveOffsets:
    z_offset = oscillation_motion(t_beats, amplitude_m, cycles_per_beat, phase_offset, waveform)
    antennas = AVAILABLE_ANTENNA_MOVES.get(antenna_move_name, move_antenna_none)(t_beats)
    return MoveOffsets(np.array([0.0, 0.0, z_offset]), np.zeros(3), antennas)

def move_side_to_side_sway(t_beats: float, amplitude_m: float = 0.03, cycles_per_beat: float = 1.0, phase_offset: float = 0.0, waveform: str = 'sin', antenna_move_name: str = 'wiggle') -> MoveOffsets:
    y_offset = oscillation_motion(t_beats, amplitude_m, cycles_per_beat, phase_offset, waveform)
    antennas = AVAILABLE_ANTENNA_MOVES.get(antenna_move_name, move_antenna_none)(t_beats)
    return MoveOffsets(np.array([0.0, y_offset, 0.0]), np.zeros(3), antennas)

def move_head_tilt_roll(t_beats: float, amplitude_rad: float = np.deg2rad(20), cycles_per_beat: float = 1.0, phase_offset: float = 0.0, waveform: str = 'sin', antenna_move_name: str = 'wiggle') -> MoveOffsets:
    roll = oscillation_motion(t_beats, amplitude_rad, cycles_per_beat, phase_offset, waveform)
    antennas = AVAILABLE_ANTENNA_MOVES.get(antenna_move_name, move_antenna_none)(t_beats)
    return MoveOffsets(np.zeros(3), np.array([roll, 0.0, 0.0]), antennas)

def move_look_around_yaw(t_beats: float, amplitude_rad: float = np.deg2rad(25), cycles_per_beat: float = 1.0, phase_offset: float = 0.0, waveform: str = 'sin', antenna_move_name: str = 'wiggle') -> MoveOffsets:
    yaw = oscillation_motion(t_beats, amplitude_rad, cycles_per_beat, phase_offset, waveform)
    antennas = AVAILABLE_ANTENNA_MOVES.get(antenna_move_name, move_antenna_none)(t_beats)
    return MoveOffsets(np.zeros(3), np.array([0.0, 0.0, yaw]), antennas)

# --- REFACTORED COMPLEX MOVES ---

def move_circular_head_roll(
    t_beats: float,
    pitch_amplitude_rad: float = np.deg2rad(10),
    roll_amplitude_rad: float = np.deg2rad(10),
    cycles_per_beat: float = 1.0,
    phase_offset: float = 0.0,
    waveform: str = 'sin',
    antenna_move_name: str = 'wiggle'
) -> MoveOffsets:
    """
    Rolls the head in a circular path using any waveform.
    This is achieved by applying a 90-degree (0.25 cycle) phase shift
    between the pitch and roll components.
    """
    # For a circular motion, one axis must lag the other by a quarter cycle.
    # We apply the user-selected waveform to both, but shift the phase of one.
    # sin(t) and cos(t) are equivalent to sin(t) and sin(t + pi/2).
    roll_offset = oscillation_motion(t_beats, roll_amplitude_rad, cycles_per_beat, phase_offset, waveform)
    pitch_offset = oscillation_motion(t_beats, pitch_amplitude_rad, cycles_per_beat, phase_offset + 0.25, waveform)

    antennas_offset = AVAILABLE_ANTENNA_MOVES.get(antenna_move_name, move_antenna_none)(t_beats)
    return MoveOffsets(np.zeros(3), np.array([roll_offset, pitch_offset, 0.0]), antennas_offset)

def move_robot_z_bounce(t_beats: float, amplitude_m: float = 0.03, cycles_per_beat: float = 1.0, phase_offset: float = 0.0, waveform: str = 'sin', antenna_move_name: str = 'wiggle') -> MoveOffsets:
    z_offset = oscillation_motion(t_beats, amplitude_m, cycles_per_beat, phase_offset, waveform)
    antennas = AVAILABLE_ANTENNA_MOVES.get(antenna_move_name, move_antenna_none)(t_beats)
    return MoveOffsets(np.array([0.0, 0.0, z_offset]), np.zeros(3), antennas)

def move_floating_x_glide(t_beats: float, amplitude_m: float = 0.04, cycles_per_beat: float = 1.0, phase_offset: float = 0.0, waveform: str = 'sin', antenna_move_name: str = 'wiggle') -> MoveOffsets:
    x_offset = oscillation_motion(t_beats, amplitude_m, cycles_per_beat, phase_offset, waveform)
    antennas = AVAILABLE_ANTENNA_MOVES.get(antenna_move_name, move_antenna_none)(t_beats)
    return MoveOffsets(np.array([x_offset, 0.0, 0.0]), np.zeros(3), antennas)

def move_side_to_side_head_bob_what_is_love(t_beats: float, roll_amplitude_rad: float = np.deg2rad(40), cycles_per_beat: float = 1.0, phase_offset: float = 0.0, waveform: str = 'sin', antenna_move_name: str = 'wiggle') -> MoveOffsets:
    roll = oscillation_motion(t_beats, roll_amplitude_rad, cycles_per_beat, phase_offset, waveform)
    antennas = AVAILABLE_ANTENNA_MOVES.get(antenna_move_name, move_antenna_none)(t_beats)
    return MoveOffsets(np.zeros(3), np.array([roll, 0.0, 0.0]), antennas)

def move_figure_eight(
    t_beats: float,
    yaw_amplitude_rad: float = np.deg2rad(20),
    pitch_amplitude_rad: float = np.deg2rad(15),
    cycles_per_beat: float = 0.25,
    phase_offset: float = 0.0,
    waveform: str = 'sin',
    antenna_move_name: str = 'wiggle'
) -> MoveOffsets:
    """
    Moves the head in a horizontal figure-eight pattern (Lissajous curve).
    This is achieved with a 1:2 frequency ratio between yaw and pitch.
    The user-selected waveform is applied to both components.
    """
    # Yaw moves at the base frequency, pitch moves at double the frequency.
    yaw_offset = oscillation_motion(t_beats, yaw_amplitude_rad, cycles_per_beat, phase_offset, waveform)
    pitch_offset = oscillation_motion(t_beats, pitch_amplitude_rad, cycles_per_beat * 2, phase_offset, waveform)

    antennas_offset = AVAILABLE_ANTENNA_MOVES.get(antenna_move_name, move_antenna_none)(t_beats)
    return MoveOffsets(np.zeros(3), np.array([0.0, pitch_offset, yaw_offset]), antennas_offset)


# --- Library of Available Moves ---
AVAILABLE_DANCE_MOVES: dict[str, Callable] = {
    "simple_nod": move_simple_nod,
    "head_bob_z": move_head_bob_z,
    "side_to_side_sway": move_side_to_side_sway,
    "head_tilt_roll": move_head_tilt_roll,
    "look_around_yaw": move_look_around_yaw,
    "circular_head_roll": move_circular_head_roll,
    "robot_z_bounce": move_robot_z_bounce,
    "floating_x_glide": move_floating_x_glide,
    "side_to_side_head_bob_what_is_love": move_side_to_side_head_bob_what_is_love,
    "figure_eight": move_figure_eight,
}

# --- Utility Functions ---
# (This section is unchanged)
def combine_offsets(*offsets: MoveOffsets) -> MoveOffsets:
    """Combines multiple MoveOffsets by summing their components."""
    if not offsets:
        return MoveOffsets(np.zeros(3), np.zeros(3), np.zeros(2))
        
    pos = sum((o.position_offset for o in offsets), np.zeros(3))
    ori = sum((o.orientation_offset for o in offsets), np.zeros(3))
    ant = sum((o.antennas_offset for o in offsets), np.zeros(2))
    return MoveOffsets(pos, ori, ant)