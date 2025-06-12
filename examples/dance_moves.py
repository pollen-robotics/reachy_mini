import numpy as np

# --- Antenna-only Move Functions ---
def move_antenna_wiggle(musical_beat_time, amplitude_rad=0.7, frequency_factor=0.5):
    """Antennas wiggle in opposition. One full cycle every 2 beats."""
    val = amplitude_rad * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    return np.array([val, -val])


def move_antenna_both_forward_back(musical_beat_time, amplitude_rad=0.6, frequency_factor=1.0):
    """Both antennas move forward and back together."""
    val = amplitude_rad * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    return np.array([val, val])

def move_antenna_none(musical_beat_time):
    """No antenna movement."""
    return np.zeros(2)

AVAILABLE_ANTENNA_MOVES = {
    "wiggle": move_antenna_wiggle,
    "both_forward_back": move_antenna_both_forward_back,
    "none": move_antenna_none,
}

# --- Individual Move Parameters ---
MOVE_SPECIFIC_PARAMS = {
    "simple_nod": {"amplitude_rad": np.deg2rad(20), "frequency_factor": 1.0, "antenna_move_name": "wiggle"},
    "head_bob_z": {"amplitude_m": 0.03, "frequency_factor": 1.0, "antenna_move_name": "both_forward_back"},
    "side_to_side_sway": {"amplitude_m": 0.04, "frequency_factor": 0.5, "antenna_move_name": "wiggle"},
    "head_tilt_roll": {"amplitude_rad": np.deg2rad(25), "frequency_factor": 0.5, "antenna_move_name": "wiggle"},
    "look_around_yaw": {"amplitude_rad": np.deg2rad(30), "frequency_factor": 0.25, "antenna_move_name": "none"},
    "circular_head_roll": {"pitch_amplitude_rad": np.deg2rad(15), "roll_amplitude_rad": np.deg2rad(15), "frequency_factor": 0.25, "antenna_move_name": "both_forward_back"},
    "robot_z_bounce": {"amplitude_m": 0.03, "frequency_factor": 0.5, "antenna_move_name": "wiggle"},
    "floating_x_glide": {"amplitude_m": 0.02, "frequency_factor": 1.0, "antenna_move_name": "wiggle"},
    "side_to_side_head_bob_what_is_love": {"roll_amplitude_rad": np.deg2rad(20), "frequency_factor": 1.0, "antenna_move_name": "wiggle"},
    "figure_eight": {"yaw_amplitude_rad": np.deg2rad(20), "pitch_amplitude_rad": np.deg2rad(15), "frequency_factor": 0.25, "antenna_move_name": "wiggle"},
}

def get_move_parameters(move_name):
    return MOVE_SPECIFIC_PARAMS.get(move_name, {})

# --- Dance Move Functions ---
def move_simple_nod(musical_beat_time, amplitude_rad=np.deg2rad(15), frequency_factor=1.0, antenna_move_name='wiggle'):
    pitch_offset = amplitude_rad * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    antennas_offset = AVAILABLE_ANTENNA_MOVES.get(antenna_move_name, move_antenna_none)(musical_beat_time)
    return {"position_offset": np.zeros(3), "orientation_offset": np.array([0.0, pitch_offset, 0.0]), "antennas_offset": antennas_offset}

def move_head_bob_z(musical_beat_time, amplitude_m=0.02, frequency_factor=1.0, antenna_move_name='wiggle'):
    z_offset = -amplitude_m * (1 - np.cos(2 * np.pi * frequency_factor * musical_beat_time)) / 2
    antennas_offset = AVAILABLE_ANTENNA_MOVES.get(antenna_move_name, move_antenna_none)(musical_beat_time)
    return {"position_offset": np.array([0.0, 0.0, z_offset]), "orientation_offset": np.zeros(3), "antennas_offset": antennas_offset}

def move_side_to_side_sway(musical_beat_time, amplitude_m=0.03, frequency_factor=1.0, antenna_move_name='wiggle'):
    y_offset = amplitude_m * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    antennas_offset = AVAILABLE_ANTENNA_MOVES.get(antenna_move_name, move_antenna_none)(musical_beat_time)
    return {"position_offset": np.array([0.0, y_offset, 0.0]), "orientation_offset": np.zeros(3), "antennas_offset": antennas_offset}

def move_head_tilt_roll(musical_beat_time, amplitude_rad=np.deg2rad(20), frequency_factor=1.0, antenna_move_name='wiggle'):
    roll_offset = amplitude_rad * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    antennas_offset = AVAILABLE_ANTENNA_MOVES.get(antenna_move_name, move_antenna_none)(musical_beat_time)
    return {"position_offset": np.zeros(3), "orientation_offset": np.array([roll_offset, 0.0, 0.0]), "antennas_offset": antennas_offset}

def move_look_around_yaw(musical_beat_time, amplitude_rad=np.deg2rad(25), frequency_factor=1.0, antenna_move_name='wiggle'):
    yaw_offset = amplitude_rad * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    antennas_offset = AVAILABLE_ANTENNA_MOVES.get(antenna_move_name, move_antenna_none)(musical_beat_time)
    return {"position_offset": np.zeros(3), "orientation_offset": np.array([0.0, 0.0, yaw_offset]), "antennas_offset": antennas_offset}

def move_circular_head_roll(musical_beat_time, pitch_amplitude_rad=np.deg2rad(10), roll_amplitude_rad=np.deg2rad(10), frequency_factor=1.0, antenna_move_name='wiggle'):
    pitch_offset = pitch_amplitude_rad * np.cos(2 * np.pi * frequency_factor * musical_beat_time)
    roll_offset = roll_amplitude_rad * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    antennas_offset = AVAILABLE_ANTENNA_MOVES.get(antenna_move_name, move_antenna_none)(musical_beat_time)
    return {"position_offset": np.zeros(3), "orientation_offset": np.array([roll_offset, pitch_offset, 0.0]), "antennas_offset": antennas_offset}

def move_robot_z_bounce(musical_beat_time, amplitude_m=0.03, frequency_factor=1.0, antenna_move_name='wiggle'):
    z_offset = amplitude_m * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    antennas_offset = AVAILABLE_ANTENNA_MOVES.get(antenna_move_name, move_antenna_none)(musical_beat_time)
    return {"position_offset": np.array([0.0, 0.0, z_offset]), "orientation_offset": np.zeros(3), "antennas_offset": antennas_offset}

def move_floating_x_glide(musical_beat_time, amplitude_m=0.04, frequency_factor=1.0, antenna_move_name='wiggle'):
    x_offset = amplitude_m * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    antennas_offset = AVAILABLE_ANTENNA_MOVES.get(antenna_move_name, move_antenna_none)(musical_beat_time)
    return {"position_offset": np.array([x_offset, 0.0, 0.0]), "orientation_offset": np.zeros(3), "antennas_offset": antennas_offset}

def move_side_to_side_head_bob_what_is_love(musical_beat_time, roll_amplitude_rad=np.deg2rad(40), frequency_factor=1.0, antenna_move_name='wiggle'):
    roll_offset = roll_amplitude_rad * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    antennas_offset = AVAILABLE_ANTENNA_MOVES.get(antenna_move_name, move_antenna_none)(musical_beat_time)
    return {"position_offset": np.zeros(3), "orientation_offset": np.array([roll_offset, 0.0, 0.0]), "antennas_offset": antennas_offset}

def move_figure_eight(musical_beat_time, yaw_amplitude_rad=np.deg2rad(20), pitch_amplitude_rad=np.deg2rad(15), frequency_factor=0.25, antenna_move_name='wiggle'):
    """Moves the head in a horizontal figure-eight pattern."""
    yaw_offset = yaw_amplitude_rad * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    pitch_offset = pitch_amplitude_rad * np.sin(2 * np.pi * 2 * frequency_factor * musical_beat_time)
    antennas_offset = AVAILABLE_ANTENNA_MOVES.get(antenna_move_name, move_antenna_none)(musical_beat_time)
    return {"position_offset": np.zeros(3), "orientation_offset": np.array([0.0, pitch_offset, yaw_offset]), "antennas_offset": antennas_offset}


# --- Library of Available Moves ---
AVAILABLE_DANCE_MOVES = {
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