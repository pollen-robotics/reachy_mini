"""Hook system for motion events via entry points.

This module provides a plugin system that allows external packages to register
callbacks for move lifecycle events (start/end) without modifying reachy_mini code.

External packages register hooks in their pyproject.toml:
    [project.entry-points."reachy_mini.motion.hooks"]
    my_plugin = "my_package:MyHook"

Example hook implementation:
    class MyHook:
        def on_move_start(self, move_name: str | None, dataset: str | None) -> None:
            print(f"Move {move_name} from {dataset} starting")
        
        def on_move_end(self, move_name: str | None, dataset: str | None) -> None:
            print(f"Move {move_name} completed")
"""

import logging
from importlib.metadata import entry_points
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class MoveHook(Protocol):
    """Protocol defining the interface that motion hooks must implement.
    
    Plugins should implement both methods to receive notifications when
    moves (emotions, dances, etc.) start and end.
    """

    def on_move_start(self, move_name: str | None, dataset: str | None) -> None:
        """Called when a move starts playing.
        
        Args:
            move_name: Name/description of the move (e.g., "happy", "confused")
            dataset: HuggingFace dataset name (e.g., "pollen-robotics/reachy-mini-emotions-library")
        """
        ...

    def on_move_end(self, move_name: str | None, dataset: str | None) -> None:
        """Called when a move finishes playing.
        
        Args:
            move_name: Name/description of the move
            dataset: HuggingFace dataset name
        """
        ...


# Global registry of loaded hooks
_hooks: list[MoveHook] = []
_loaded = False


def load_hooks() -> None:
    """Discover and load all registered motion hooks via entry points.
    
    This function is called automatically during reachy_mini initialization.
    Hooks are discovered from the "reachy_mini.motion.hooks" entry point group.
    
    If a hook fails to load, a warning is logged but execution continues.
    """
    global _loaded
    if _loaded:
        return
    
    # Discover all registered hooks via entry points
    eps = entry_points(group="reachy_mini.motion.hooks")
    
    for ep in eps:
        try:
            # Load the hook class and instantiate it
            hook_class = ep.load()
            hook_instance = hook_class()
            _hooks.append(hook_instance)
            logger.info(f"Loaded motion hook: {ep.name} from {ep.value}")
        except Exception as e:
            logger.warning(f"Failed to load motion hook {ep.name}: {e}")
    
    _loaded = True
    if _hooks:
        logger.info(f"Loaded {len(_hooks)} motion hook(s)")


def notify_start(move_name: str | None = None, dataset: str | None = None) -> None:
    """Notify all registered hooks that a move is starting.
    
    Args:
        move_name: Name/description of the move being played
        dataset: HuggingFace dataset name the move comes from
    """
    for hook in _hooks:
        try:
            hook.on_move_start(move_name, dataset)
        except Exception as e:
            # Don't let plugin errors crash the main application
            logger.debug(f"Hook error in on_move_start: {e}")


def notify_end(move_name: str | None = None, dataset: str | None = None) -> None:
    """Notify all registered hooks that a move has ended.
    
    Args:
        move_name: Name/description of the move that finished
        dataset: HuggingFace dataset name the move came from
    """
    for hook in _hooks:
        try:
            hook.on_move_end(move_name, dataset)
        except Exception as e:
            # Don't let plugin errors crash the main application
            logger.debug(f"Hook error in on_move_end: {e}")
