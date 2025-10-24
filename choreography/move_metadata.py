"""
Move Metadata: Beat counts and characteristics for choreography generation
Beat count = number of distinct movement phrases in the move
"""

# Dance moves with their beat counts
DANCE_BEAT_COUNTS = {
    # Based on move descriptions and rhythmic structure
    "simple_nod": 1,                  # Single up-down nod
    "side_to_side_sway": 2,           # Left sway, right sway
    "pendulum_swing": 2,              # Swing left, swing right
    "stumble_and_recover": 2,         # Stumble, recover
    "side_peekaboo": 2,               # Peek left, peek right
    "head_tilt_roll": 2,              # Tilt, roll
    "uh_huh_tilt": 1,                 # Single affirmative motion
    "yeah_nod": 1,                    # Single emphatic nod
    "neck_recoil": 1,                 # Single recoil motion
    "chicken_peck": 1,                # Single peck motion
    "groovy_sway_and_roll": 4,        # Sway left, sway right, roll left, roll right
    "jackson_square": 4,              # 4 corners of square pattern
    "chin_lead": 2,                   # Lead forward, return
    "head_tilt_roll": 2,              # Tilt one way, tilt other
    "grid_snap": 4,                   # 4 grid positions
    "polyrhythm_combo": 4,            # Multiple overlapping rhythms
    "interwoven_spirals": 4,          # Complex multi-beat pattern
    "headbanger_combo": 2,            # Bang down, bang up
    "dizzy_spin": 4,                  # 4 quarter rotations
    "sharp_side_tilt": 2,             # Sharp left, sharp right
}

# Emotion moves - typically 1 beat (single expression/gesture)
EMOTION_BEAT_COUNTS = {
    # Most emotions are single poses/gestures = 1 beat
    # This is a reasonable default for all emotions
}

def get_beat_count(move_name: str, move_type: str = "dance") -> int:
    """
    Get the beat count for a move.

    Args:
        move_name: Name of the move
        move_type: "dance" or "emotion"

    Returns:
        Beat count (default 1 for emotions, must be defined for dances)
    """
    if move_type == "emotion":
        # Emotions are typically single expressions = 1 beat
        return EMOTION_BEAT_COUNTS.get(move_name, 1)
    else:
        # Dances should be explicitly defined
        return DANCE_BEAT_COUNTS.get(move_name, 2)  # Default to 2 if unknown

def calculate_move_duration(move_name: str, move_type: str, cycles: int, bpm: float) -> float:
    """
    Calculate actual playback duration using BPM timing formula.

    Formula: Move Duration = cycles × (intrinsic beat count) × (60/BPM)

    Args:
        move_name: Name of the move
        move_type: "dance" or "emotion"
        cycles: Number of repetitions
        bpm: Beats per minute

    Returns:
        Duration in seconds
    """
    beat_count = get_beat_count(move_name, move_type)
    beat_duration = 60.0 / bpm
    return cycles * beat_count * beat_duration
