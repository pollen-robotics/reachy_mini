"""
Move Metadata Cache System

Builds and maintains a cache of move metadata (actual SDK durations, types, etc.)
This is expensive to build but fast to load once cached.
"""

import json
import os
from pathlib import Path
from reachy_mini.motion.recorded_move import RecordedMoves

CACHE_FILE = Path(__file__).parent / "move_metadata.json"

DANCE_LIBRARY = "pollen-robotics/reachy-mini-dances-library"
EMOTION_LIBRARY = "pollen-robotics/reachy-mini-emotions-library"

# Moves that cause mechanical collisions on physical Reachy Mini hardware
FORBIDDEN_MOVES = {
    'headbanger_combo', # Excessive strain, collision risk
    'grid_snap',        # Sharp movements cause internal interference
    'chin_lead',        # Head position conflicts with internal components
    'dying1'            # Extended sequence with collision points
}


def build_cache():
    """
    Build the move metadata cache by loading all moves from SDK.
    This is slow (loads from HuggingFace) but only needs to run once.
    """
    print("[MoveCache] Building move metadata cache...")
    print("[MoveCache] This will take ~30s (loading from HuggingFace)...")

    metadata = {}

    # Load dance library
    print(f"[MoveCache] Loading {DANCE_LIBRARY}...")
    try:
        dances = RecordedMoves(DANCE_LIBRARY)
        dance_names = list(dances.moves.keys())

        for name in dance_names:
            if name in FORBIDDEN_MOVES:
                print(f"[MoveCache] Skipping forbidden move: {name}")
                continue
            move = dances.get(name)
            metadata[name] = {
                'duration': float(move.duration),
                'type': 'dance',
                'library': DANCE_LIBRARY
            }

        print(f"[MoveCache] Loaded {len(dance_names)} dance moves")
    except Exception as e:
        print(f"[MoveCache] Error loading dance library: {e}")

    # Load emotion library
    print(f"[MoveCache] Loading {EMOTION_LIBRARY}...")
    try:
        emotions = RecordedMoves(EMOTION_LIBRARY)
        emotion_names = list(emotions.moves.keys())

        for name in emotion_names:
            if name in FORBIDDEN_MOVES:
                print(f"[MoveCache] Skipping forbidden move: {name}")
                continue
            move = emotions.get(name)
            metadata[name] = {
                'duration': float(move.duration),
                'type': 'emotion',
                'library': EMOTION_LIBRARY
            }

        print(f"[MoveCache] Loaded {len(emotion_names)} emotion moves")
    except Exception as e:
        print(f"[MoveCache] Error loading emotion library: {e}")

    # Save to cache
    print(f"[MoveCache] Saving cache to {CACHE_FILE}...")
    with open(CACHE_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[MoveCache] Cache built successfully! Total moves: {len(metadata)}")
    return metadata


def load_cache(rebuild=False):
    """
    Load move metadata from cache.
    If cache doesn't exist or rebuild=True, builds it first.

    Returns:
        dict: {move_name: {duration: float, type: str, library: str}}
    """
    if rebuild or not CACHE_FILE.exists():
        return build_cache()

    print(f"[MoveCache] Loading cache from {CACHE_FILE}...")
    with open(CACHE_FILE, 'r') as f:
        metadata = json.load(f)

    print(f"[MoveCache] Loaded {len(metadata)} moves from cache")
    return metadata


def get_move_duration(move_name, metadata=None):
    """
    Get the actual SDK duration for a move.

    Args:
        move_name: Name of the move
        metadata: Optional pre-loaded metadata dict (loads cache if not provided)

    Returns:
        float: Duration in seconds, or None if move not found
    """
    if metadata is None:
        metadata = load_cache()

    move_data = metadata.get(move_name)
    if move_data:
        return move_data['duration']
    return None


def get_moves_by_type(move_type, metadata=None):
    """
    Get all moves of a specific type.

    Args:
        move_type: "dance" or "emotion"
        metadata: Optional pre-loaded metadata dict

    Returns:
        list: Move names of the specified type
    """
    if metadata is None:
        metadata = load_cache()

    return [name for name, data in metadata.items() if data['type'] == move_type]


def get_moves_by_duration(min_dur, max_dur, metadata=None):
    """
    Get moves within a duration range.

    Args:
        min_dur: Minimum duration in seconds
        max_dur: Maximum duration in seconds
        metadata: Optional pre-loaded metadata dict

    Returns:
        list: Move names within the duration range
    """
    if metadata is None:
        metadata = load_cache()

    return [name for name, data in metadata.items()
            if min_dur <= data['duration'] <= max_dur]


def get_move_info(move_name, metadata=None):
    """
    Get full metadata for a move.

    Args:
        move_name: Name of the move
        metadata: Optional pre-loaded metadata dict

    Returns:
        dict: {duration: float, type: str, library: str} or None
    """
    if metadata is None:
        metadata = load_cache()

    return metadata.get(move_name)


if __name__ == "__main__":
    # Build/rebuild cache
    import sys
    rebuild = "--rebuild" in sys.argv

    metadata = load_cache(rebuild=rebuild)

    # Print summary
    dances = get_moves_by_type("dance", metadata)
    emotions = get_moves_by_type("emotion", metadata)

    print("\n" + "="*60)
    print("MOVE METADATA CACHE SUMMARY")
    print("="*60)
    print(f"Total moves: {len(metadata)}")
    print(f"Dance moves: {len(dances)}")
    print(f"Emotion moves: {len(emotions)}")
    print(f"\nDuration range:")
    durations = [data['duration'] for data in metadata.values()]
    print(f"  Min: {min(durations):.3f}s")
    print(f"  Max: {max(durations):.3f}s")
    print(f"  Average: {sum(durations)/len(durations):.3f}s")
    print("="*60)
