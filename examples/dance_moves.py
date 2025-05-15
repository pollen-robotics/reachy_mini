import numpy as np

# --- Individual Move Parameters ---
# Names are now lowercase_with_underscores
MOVE_SPECIFIC_PARAMS = {
    "simple_nod": {"amplitude_rad": np.deg2rad(20), "frequency_factor": 1.0},
    "head_bob_z": {"amplitude_m": 0.03, "frequency_factor": 1.0},
    "side_to_side_sway": {"amplitude_m": 0.04, "frequency_factor": 0.5},
    "head_tilt_roll": {"amplitude_rad": np.deg2rad(25), "frequency_factor": 0.5},
    "look_around_yaw": {"amplitude_rad": np.deg2rad(30), "frequency_factor": 0.25},
    "circular_head_roll": {"pitch_amplitude_rad": np.deg2rad(15), "roll_amplitude_rad": np.deg2rad(15), "frequency_factor": 0.25},
    "robot_z_bounce": {"amplitude_m": 0.035, "frequency_factor": 2.0},
    "floating_x_glide": {"amplitude_m": 0.05, "frequency_factor": 1.0},
    "side_to_side_head_bob_what_is_love": {"roll_amplitude_rad": np.deg2rad(40), "frequency_factor": 1.0}
}

def get_move_parameters(move_name):
    return MOVE_SPECIFIC_PARAMS.get(move_name, {})

# --- Dance Move Functions ---

def move_simple_nod(musical_beat_time, amplitude_rad=np.deg2rad(15), frequency_factor=1.0):
    pitch_offset = amplitude_rad * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    return {"position_offset": np.zeros(3), "orientation_offset": np.array([0.0, pitch_offset, 0.0])}

def move_head_bob_z(musical_beat_time, amplitude_m=0.02, frequency_factor=1.0):
    z_offset = -amplitude_m * (1 - np.cos(2 * np.pi * frequency_factor * musical_beat_time)) / 2
    return {"position_offset": np.array([0.0, 0.0, z_offset]), "orientation_offset": np.zeros(3)}

def move_side_to_side_sway(musical_beat_time, amplitude_m=0.03, frequency_factor=1.0):
    y_offset = amplitude_m * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    return {"position_offset": np.array([0.0, y_offset, 0.0]), "orientation_offset": np.zeros(3)}

def move_head_tilt_roll(musical_beat_time, amplitude_rad=np.deg2rad(20), frequency_factor=1.0):
    roll_offset = amplitude_rad * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    return {"position_offset": np.zeros(3), "orientation_offset": np.array([roll_offset, 0.0, 0.0])}

def move_look_around_yaw(musical_beat_time, amplitude_rad=np.deg2rad(25), frequency_factor=1.0):
    yaw_offset = amplitude_rad * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    return {"position_offset": np.zeros(3), "orientation_offset": np.array([0.0, 0.0, yaw_offset])}

def move_circular_head_roll(musical_beat_time, pitch_amplitude_rad=np.deg2rad(10), roll_amplitude_rad=np.deg2rad(10), frequency_factor=1.0):
    pitch_offset = pitch_amplitude_rad * np.cos(2 * np.pi * frequency_factor * musical_beat_time)
    roll_offset = roll_amplitude_rad * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    return {"position_offset": np.zeros(3), "orientation_offset": np.array([roll_offset, pitch_offset, 0.0])}

def move_robot_z_bounce(musical_beat_time, amplitude_m=0.03, frequency_factor=1.0):
    z_offset = amplitude_m * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    return {"position_offset": np.array([0.0, 0.0, z_offset]), "orientation_offset": np.zeros(3)}

def move_floating_x_glide(musical_beat_time, amplitude_m=0.04, frequency_factor=1.0):
    x_offset = amplitude_m * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    return {"position_offset": np.array([x_offset, 0.0, 0.0]), "orientation_offset": np.zeros(3)}

def move_side_to_side_head_bob_what_is_love(musical_beat_time, roll_amplitude_rad=np.deg2rad(40), frequency_factor=1.0):
    """Mimics the classic head bob from 'What is Love'. This is primarily a roll motion.
       frequency_factor = 1.0 means one full left-right-center cycle per beat of the music.
    """
    roll_offset = roll_amplitude_rad * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    return {
        "position_offset": np.zeros(3),
        "orientation_offset": np.array([roll_offset, 0.0, 0.0])
    }


# --- Library of Available Moves ---
# Keys are now lowercase_with_underscores
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
}