"""Motion module for Reachy Mini.

This module contains both utilities to create and play moves, as well as utilities to download datasets of recorded moves.

For plugin authors: import MoveHook to create motion event hooks.
"""

from reachy_mini.motion.hooks import MoveHook, load_hooks

__all__ = ["MoveHook", "load_hooks"]
