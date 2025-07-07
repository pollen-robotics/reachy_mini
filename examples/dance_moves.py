"""
Dance Motion Library (v3.1)
---------------------------
A rich, compositional library of beat-synchronized motion primitives.

This version introduces a completely rebuilt 'side_peekaboo' with a full
left-and-right sequence, and adds new characterful moves like the 'chicken_peck'
and the compositional 'groovy_sway_and_roll'.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable

# ... (The MoveOffsets class and motion primitives are unchanged) ...
@dataclass
class MoveOffsets:
    position_offset: np.ndarray
    orientation_offset: np.ndarray
    antennas_offset: np.ndarray

def oscillation_motion(t_beats: float, amplitude: float, cycles_per_beat: float = 1.0, phase_offset: float = 0.0, waveform: str = 'sin') -> float:
    phase = 2 * np.pi * (cycles_per_beat * t_beats + phase_offset)
    if waveform == 'sin': return amplitude * np.sin(phase)
    elif waveform == 'cos': return amplitude * np.cos(phase)
    elif waveform == 'square': return amplitude * np.sign(np.sin(phase))
    elif waveform == 'triangle': return amplitude * (2 / np.pi) * np.arcsin(np.sin(phase))
    elif waveform == 'sawtooth': return amplitude * (2 * ((phase / (2 * np.pi)) % 1) - 1)
    else: raise ValueError(f"Unsupported waveform type: {waveform}")

def transient_motion(t_beats: float, amplitude: float, duration_beats: float = 1.0, delay_beats: float = 0.0, repeat_every: float = 0.0):
    if repeat_every <= 0.0: repeat_every = duration_beats + delay_beats
    start_time = np.floor((t_beats - delay_beats) / repeat_every) * repeat_every + delay_beats
    if start_time <= t_beats < start_time + duration_beats:
        t_norm = (t_beats - start_time) / duration_beats
        eased_t = t_norm * t_norm * (3.0 - 2.0 * t_norm)
        return amplitude * eased_t
    return 0.0

def combine_offsets(*offsets: MoveOffsets) -> MoveOffsets:
    if not offsets: return MoveOffsets(np.zeros(3), np.zeros(3), np.zeros(2))
    pos = sum((o.position_offset for o in offsets), np.zeros(3))
    ori = sum((o.orientation_offset for o in offsets), np.zeros(3))
    ant = sum((o.antennas_offset for o in offsets), np.zeros(2))
    return MoveOffsets(pos, ori, ant)

# ────────────────────────────── ATOMIC MOVES & HELPERS ──────────────────────────────────
# (This section is unchanged from the previous working version)
def atomic_x_pos(t_beats, **kwargs): return MoveOffsets(np.array([oscillation_motion(t_beats, **kwargs), 0, 0]), np.zeros(3), np.zeros(2))
def atomic_y_pos(t_beats, **kwargs): return MoveOffsets(np.array([0, oscillation_motion(t_beats, **kwargs), 0]), np.zeros(3), np.zeros(2))
def atomic_z_pos(t_beats, **kwargs): return MoveOffsets(np.array([0, 0, oscillation_motion(t_beats, **kwargs)]), np.zeros(3), np.zeros(2))
def atomic_roll(t_beats, **kwargs): return MoveOffsets(np.zeros(3), np.array([oscillation_motion(t_beats, **kwargs), 0, 0]), np.zeros(2))
def atomic_pitch(t_beats, **kwargs): return MoveOffsets(np.zeros(3), np.array([0, oscillation_motion(t_beats, **kwargs), 0]), np.zeros(2))
def atomic_yaw(t_beats, **kwargs): return MoveOffsets(np.zeros(3), np.array([0, 0, oscillation_motion(t_beats, **kwargs)]), np.zeros(2))
def atomic_antenna_wiggle(t_beats, **kwargs): return MoveOffsets(np.zeros(3), np.zeros(3), np.array([oscillation_motion(t_beats, **kwargs), -oscillation_motion(t_beats, **kwargs)]))
def atomic_antenna_both(t_beats, **kwargs): return MoveOffsets(np.zeros(3), np.zeros(3), np.array([oscillation_motion(t_beats, **kwargs)]*2))
AVAILABLE_ANTENNA_MOVES = { "wiggle": atomic_antenna_wiggle, "both": atomic_antenna_both }
VALID_OSCILLATION_KEYS = {'cycles_per_beat', 'phase_offset', 'waveform'}
def get_base_move_kwargs(main_kwargs): return {key: main_kwargs[key] for key in VALID_OSCILLATION_KEYS if key in main_kwargs}
def get_antenna_kwargs(main_kwargs):
    antenna_params = get_base_move_kwargs(main_kwargs)
    if 'antenna_amplitude_rad' in main_kwargs: antenna_params['amplitude'] = main_kwargs['antenna_amplitude_rad']
    return antenna_params

# ──────────────────── COMPOSITE & CHOREOGRAPHED MOVES ──────────────────────

def move_simple_nod(t_beats, amplitude_rad, antenna_move_name='wiggle', **kwargs):
    base_move = atomic_pitch(t_beats, amplitude=amplitude_rad, **get_base_move_kwargs(kwargs))
    antenna_move = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, **get_antenna_kwargs(kwargs))
    return combine_offsets(base_move, antenna_move)

def move_head_tilt_roll(t_beats, amplitude_rad, antenna_move_name='wiggle', **kwargs):
    base_move = atomic_roll(t_beats, amplitude=amplitude_rad, **get_base_move_kwargs(kwargs))
    antenna_move = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, **get_antenna_kwargs(kwargs))
    return combine_offsets(base_move, antenna_move)

def move_dizzy_spin(t_beats, roll_amplitude_rad, pitch_amplitude_rad, antenna_move_name='wiggle', **kwargs):
    base_kwargs = get_base_move_kwargs(kwargs)
    roll = atomic_roll(t_beats, amplitude=roll_amplitude_rad, **base_kwargs)
    pitch_kwargs = base_kwargs.copy(); pitch_kwargs['phase_offset'] = pitch_kwargs.get('phase_offset', 0) + 0.25
    pitch = atomic_pitch(t_beats, amplitude=pitch_amplitude_rad, **pitch_kwargs)
    antenna_move = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, **get_antenna_kwargs(kwargs))
    return combine_offsets(roll, pitch, antenna_move)

def move_no_shake(t_beats, amplitude_rad, antenna_move_name='wiggle', **kwargs):
    base_move = atomic_yaw(t_beats, amplitude=amplitude_rad, **get_base_move_kwargs(kwargs))
    antenna_move = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, **get_antenna_kwargs(kwargs))
    return combine_offsets(base_move, antenna_move)

def move_interwoven_spirals(t_beats, roll_amp, pitch_amp, yaw_amp, antenna_move_name='wiggle', **kwargs):
    base_kwargs = get_base_move_kwargs(kwargs)
    roll_kwargs = base_kwargs.copy(); roll_kwargs['cycles_per_beat'] = 0.125
    pitch_kwargs = base_kwargs.copy(); pitch_kwargs.update({'cycles_per_beat': 0.25, 'phase_offset': base_kwargs.get('phase_offset', 0) + 0.25})
    yaw_kwargs = base_kwargs.copy(); yaw_kwargs.update({'cycles_per_beat': 0.5, 'phase_offset': base_kwargs.get('phase_offset', 0) + 0.5})
    roll = atomic_roll(t_beats, amplitude=roll_amp, **roll_kwargs)
    pitch = atomic_pitch(t_beats, amplitude=pitch_amp, **pitch_kwargs)
    yaw = atomic_yaw(t_beats, amplitude=yaw_amp, **yaw_kwargs)
    antenna_move = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, **get_antenna_kwargs(kwargs))
    return combine_offsets(roll, pitch, yaw, antenna_move)

def move_indian_head_slide(t_beats, amplitude_m, antenna_move_name='wiggle', **kwargs):
    base_kwargs = get_base_move_kwargs(kwargs)
    sway = atomic_y_pos(t_beats, amplitude=amplitude_m, **base_kwargs)
    counter_roll = atomic_roll(t_beats, amplitude=-(amplitude_m * 2.5), **base_kwargs)
    antenna_move = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, **get_antenna_kwargs(kwargs))
    return combine_offsets(sway, counter_roll, antenna_move)

def move_cocky_sway(t_beats, y_amplitude_m, z_amplitude_m, antenna_move_name='wiggle', **kwargs):
    base_kwargs = get_base_move_kwargs(kwargs)
    sway = atomic_y_pos(t_beats, amplitude=y_amplitude_m, **base_kwargs)
    dip_kwargs = base_kwargs.copy(); dip_kwargs['cycles_per_beat'] = dip_kwargs.get('cycles_per_beat', 1.0) * 2
    z_offset = -z_amplitude_m * (1 + oscillation_motion(t_beats, amplitude=1, waveform='cos', **dip_kwargs)) / 2
    dip = MoveOffsets(np.array([0,0,z_offset]), np.zeros(3), np.zeros(2))
    antenna_move = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, **get_antenna_kwargs(kwargs))
    return combine_offsets(sway, dip, antenna_move)

# ─────────────────── NEW & REBUILT MOVES (v3.1) ──────────────────────

def move_side_peekaboo(t_beats, z_amp, y_amp, pitch_amp, antenna_move_name='both', **kwargs):
    """Hides down, pops up left, hides, pops up right, hides, repeat.
    Designer's Note (REBUILT): A full 16-beat sequence. Y-amplitude is tuned
    down. This is now a complete and very endearing signature move.
    """
    period = 16.0; t_in_period = t_beats % period
    pos, ori = np.zeros(3), np.zeros(3)
    def ease(t): return t*t*(3-2*t) # ease-in-out
    def excited_nod(t): return pitch_amp * np.sin(t * np.pi)

    if t_in_period < 2.0: # 1. Tuck down at center
        pos[2] = -z_amp * ease(t_in_period / 2.0)
    elif t_in_period < 4.0: # 2. Pop up and slide left
        t = (t_in_period - 2.0) / 2.0
        pos[2], pos[1], ori[1] = -z_amp * (1-ease(t)), y_amp * ease(t), excited_nod(t)
    elif t_in_period < 6.0: # 3. Tuck down on left
        t = (t_in_period - 4.0) / 2.0
        pos[2], pos[1] = -z_amp * ease(t), y_amp
    elif t_in_period < 8.0: # 4. Pop up and return to center
        t = (t_in_period - 6.0) / 2.0
        pos[2], pos[1] = -z_amp * (1-ease(t)), y_amp * (1-ease(t))
    elif t_in_period < 10.0: # 5. Tuck down at center again
        t = (t_in_period - 8.0) / 2.0
        pos[2] = -z_amp * ease(t)
    elif t_in_period < 12.0: # 6. Pop up and slide right
        t = (t_in_period - 10.0) / 2.0
        pos[2], pos[1], ori[1] = -z_amp * (1-ease(t)), -y_amp * ease(t), -excited_nod(t)
    elif t_in_period < 14.0: # 7. Tuck down on right
        t = (t_in_period - 12.0) / 2.0
        pos[2], pos[1] = -z_amp * ease(t), -y_amp
    else: # 8. Pop up and return to center
        t = (t_in_period - 14.0) / 2.0
        pos[2], pos[1] = -z_amp * (1-ease(t)), -y_amp * (1-ease(t))

    antenna_kwargs = get_antenna_kwargs(kwargs)
    antenna_kwargs['cycles_per_beat'] = 4.0 # Make antennas extra lively for this move
    antenna_move = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, **antenna_kwargs)
    return combine_offsets(MoveOffsets(pos, ori, np.zeros(2)), antenna_move)

def move_chicken_peck(t_beats, amplitude_m, antenna_move_name='both', **kwargs):
    """A series of quick, sharp, forward pecking motions.
    Designer's Note: A percussive and characterful move using the transient
    motion primitive for a non-looping feel on each beat.
    """
    base_kwargs = get_base_move_kwargs(kwargs)
    repeat_every = 1.0 / base_kwargs.get('cycles_per_beat', 1.0)
    
    x_offset = transient_motion(t_beats, amplitude_m, duration_beats=repeat_every*0.8, repeat_every=repeat_every)
    pitch_offset = transient_motion(t_beats, amplitude_m * 5, duration_beats=repeat_every*0.8, repeat_every=repeat_every)
    
    antenna_kwargs = get_antenna_kwargs(kwargs)
    antenna_kwargs['amplitude'] = antenna_kwargs.get('amplitude', 1.0) * 0.5 # Less antenna motion
    antenna_move = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, **antenna_kwargs)
    return combine_offsets(MoveOffsets(np.array([x_offset, 0, 0]), np.array([0, pitch_offset, 0]), np.zeros(2)), antenna_move)

def move_groovy_sway_and_roll(t_beats, y_amplitude_m, roll_amplitude_rad, antenna_move_name='wiggle', **kwargs):
    """A very fluid combination of a side-to-side sway and a lagging head roll.
    Designer's Note: A perfect example of how combining simple moves with a phase
    offset can create a complex, groovy, and relaxed motion.
    """
    base_kwargs = get_base_move_kwargs(kwargs)
    sway = atomic_y_pos(t_beats, amplitude=y_amplitude_m, **base_kwargs)
    
    roll_kwargs = base_kwargs.copy(); roll_kwargs['phase_offset'] = roll_kwargs.get('phase_offset', 0) + 0.25
    roll = atomic_roll(t_beats, amplitude=roll_amplitude_rad, **roll_kwargs)

    antenna_move = AVAILABLE_ANTENNA_MOVES[antenna_move_name](t_beats, **get_antenna_kwargs(kwargs))
    return combine_offsets(sway, roll, antenna_move)

# ────────────────────────── MASTER MOVE DICTIONARIES ──────────────────────────
DEFAULT_ANTENNA_PARAMS = {"antenna_move_name": "wiggle", "antenna_amplitude_rad": np.deg2rad(45)}

MOVE_SPECIFIC_PARAMS = {
    # -- Core Moves --
    "simple_nod": {"amplitude_rad": np.deg2rad(20), "cycles_per_beat": 1.0, **DEFAULT_ANTENNA_PARAMS},
    "head_tilt_roll": {"amplitude_rad": np.deg2rad(15), "cycles_per_beat": 0.5, **DEFAULT_ANTENNA_PARAMS}, # TUNED
    "dizzy_spin": {"roll_amplitude_rad": np.deg2rad(15), "pitch_amplitude_rad": np.deg2rad(15), "cycles_per_beat": 0.5, **DEFAULT_ANTENNA_PARAMS},
    "no_shake": {"amplitude_rad": np.deg2rad(25), "cycles_per_beat": 1.25, **DEFAULT_ANTENNA_PARAMS}, # TUNED
    "interwoven_spirals": {"roll_amp": np.deg2rad(15), "pitch_amp": np.deg2rad(20), "yaw_amp": np.deg2rad(25), "cycles_per_beat": 0.25, **DEFAULT_ANTENNA_PARAMS},
    # -- Creative Moves --
    "indian_head_slide": {"amplitude_m": 0.03, "cycles_per_beat": 0.5, **DEFAULT_ANTENNA_PARAMS},
    "cocky_sway": {"y_amplitude_m": 0.04, "z_amplitude_m": 0.02, "cycles_per_beat": 0.5, **DEFAULT_ANTENNA_PARAMS},
    "side_peekaboo": {"z_amp": 0.05, "y_amp": 0.03, "pitch_amp": np.deg2rad(15), "antenna_move_name": "both", "antenna_amplitude_rad": np.deg2rad(60)}, # TUNED
    "chicken_peck": {"amplitude_m": 0.02, "cycles_per_beat": 1.0, "antenna_move_name": "both", "antenna_amplitude_rad": np.deg2rad(30)},
    "groovy_sway_and_roll": {"y_amplitude_m": 0.03, "roll_amplitude_rad": np.deg2rad(15), "cycles_per_beat": 0.5, **DEFAULT_ANTENNA_PARAMS},
}

AVAILABLE_DANCE_MOVES: dict[str, Callable] = {
    # -- Core Moves --
    "simple_nod": move_simple_nod,
    "head_tilt_roll": move_head_tilt_roll,
    "dizzy_spin": move_dizzy_spin,
    "no_shake": move_no_shake,
    "interwoven_spirals": move_interwoven_spirals,
    # -- Creative Moves --
    "indian_head_slide": move_indian_head_slide,
    "cocky_sway": move_cocky_sway,
    "groovy_sway_and_roll": move_groovy_sway_and_roll,
    "chicken_peck": move_chicken_peck,
    "side_peekaboo": move_side_peekaboo,
}