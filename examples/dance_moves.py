import numpy as np

# --- Individual Move Parameters ---
# This dictionary stores the default parameters for each move.
# When a move function is called, these parameters will override the
# default arguments defined in the function's signature.
# If a move is not listed here, or a parameter for a move is not listed,
# the function's own default arguments will be used.
MOVE_SPECIFIC_PARAMS = {
    "simple_nod": {"amplitude_rad": np.deg2rad(20), "frequency_factor": 1.0},
    "head_bob_z": {"amplitude_m": 0.03, "frequency_factor": 1.0},
    "side_to_side_sway": {"amplitude_m": 0.04, "frequency_factor": 0.5},
    "head_tilt_roll": {"amplitude_rad": np.deg2rad(25), "frequency_factor": 0.5},
    "look_around_yaw": {"amplitude_rad": np.deg2rad(30), "frequency_factor": 0.25},
    "circular_head_roll": {"pitch_amplitude_rad": np.deg2rad(15), "roll_amplitude_rad": np.deg2rad(15), "frequency_factor": 0.25},
    "robot_z_bounce": {"amplitude_m": 0.03, "frequency_factor": 0.5},
    "floating_x_glide": {"amplitude_m": 0.02, "frequency_factor": 1.0},
    "side_to_side_head_bob_what_is_love": {"roll_amplitude_rad": np.deg2rad(20), "frequency_factor": 1.0}
}

def get_move_parameters(move_name):
    """
    Retrieves the specific parameters for a given move_name.
    Returns an empty dictionary if the move_name is not found in MOVE_SPECIFIC_PARAMS,
    allowing the move function to use its own default arguments.
    """
    return MOVE_SPECIFIC_PARAMS.get(move_name, {})

# --- Dance Move Functions ---
# Each function calculates position and orientation offsets based on musical_beat_time.
# - musical_beat_time: A float representing the current time in terms of musical beats.
# - Other parameters: Define the characteristics of the move (e.g., amplitude, frequency).
#   These can have default values in the function signature, which can be
#   overridden by entries in MOVE_SPECIFIC_PARAMS.
#
# Returns:
#   A dictionary: {"position_offset": np.array([dx, dy, dz]),
#                  "orientation_offset": np.array([droll, dpitch, dyaw])} # Euler XYZ in radians

def move_simple_nod(musical_beat_time, amplitude_rad=np.deg2rad(15), frequency_factor=1.0):
    pitch_offset = amplitude_rad * np.sin(2 * np.pi * frequency_factor * musical_beat_time)
    return {"position_offset": np.zeros(3), "orientation_offset": np.array([0.0, pitch_offset, 0.0])}

def move_head_bob_z(musical_beat_time, amplitude_m=0.02, frequency_factor=1.0):
    # Uses (1 - cos)/2 for a smooth 0 to -amplitude_m motion
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
    # Simple sine wave for up-down bounce around neutral
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
# This dictionary maps string names to the actual move functions.
# It's used by the main script to find and execute dances.
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

# --- A Note on Adding New Moves ---
# To add a new dance move:
# 1. Define your new move function (e.g., `def move_my_new_cool_move(musical_beat_time, param1=val1, ...):`).
#    Ensure it returns a dictionary with "position_offset" and "orientation_offset".
# 2. If you want to easily configure its default parameters (like amplitude or frequency_factor)
#    without changing the function's signature's default arguments directly, add an entry for it
#    in `MOVE_SPECIFIC_PARAMS`. The keys in this entry should match the parameter names in your function.
#    (e.g., `MOVE_SPECIFIC_PARAMS["my_new_cool_move"] = {"param1": new_default_val1}`).
# 3. Add the string name of your move and the function reference to the `AVAILABLE_DANCE_MOVES`
#    dictionary (e.g., `"my_new_cool_move": move_my_new_cool_move`).
#
# This structure (function, optional entry in MOVE_SPECIFIC_PARAMS, entry in AVAILABLE_DANCE_MOVES)
# keeps parameters configurable and allows the main script to discover and call moves by their string names.
# The `MOVE_SPECIFIC_PARAMS` provides a layer of indirection for default values, which can be useful.
# If a move is not in `MOVE_SPECIFIC_PARAMS`, or a specific parameter is not listed there for a move,
# the default values from the function's signature itself will be used.