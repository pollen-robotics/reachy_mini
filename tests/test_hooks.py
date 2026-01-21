"""Tests for the motion hooks plugin system.

These tests verify that:
1. Hooks can be loaded and discovered
2. Hooks are notified when moves start/end
3. Hook errors don't crash the main application
4. Hooks work with both RecordedMove and custom Move implementations
"""

import sys
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from reachy_mini.motion.hooks import (
    MoveHook,
    _hooks,
    load_hooks,
    notify_end,
    notify_start,
)
from reachy_mini.motion.move import Move
from reachy_mini.motion.recorded_move import RecordedMove


class MockHook:
    """Mock hook for testing."""

    def __init__(self) -> None:
        self.started_moves: list[tuple[str | None, str | None]] = []
        self.ended_moves: list[tuple[str | None, str | None]] = []

    def on_move_start(self, move_name: str | None, dataset: str | None) -> None:
        """Record that a move started."""
        self.started_moves.append((move_name, dataset))

    def on_move_end(self, move_name: str | None, dataset: str | None) -> None:
        """Record that a move ended."""
        self.ended_moves.append((move_name, dataset))


class FaultyHook:
    """Hook that raises errors for testing error handling."""

    def on_move_start(self, move_name: str | None, dataset: str | None) -> None:
        """Raise an error."""
        raise RuntimeError("Hook error in start")

    def on_move_end(self, move_name: str | None, dataset: str | None) -> None:
        """Raise an error."""
        raise RuntimeError("Hook error in end")


class DummyMove(Move):
    """Simple move implementation for testing."""

    def __init__(
        self, duration: float = 1.0, description: str | None = None, dataset_name: str | None = None
    ) -> None:
        self.description = description
        self.dataset_name = dataset_name
        self._duration = duration

    @property
    def duration(self) -> float:
        return self._duration

    def evaluate(self, t: float) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
        return None, None, None


@pytest.fixture
def reset_hooks():
    """Reset global hooks state before each test."""
    global _hooks
    import reachy_mini.motion.hooks as hooks_module

    original_hooks = hooks_module._hooks.copy()
    original_loaded = hooks_module._loaded

    hooks_module._hooks = []
    hooks_module._loaded = False

    yield

    hooks_module._hooks = original_hooks
    hooks_module._loaded = original_loaded


def test_move_hook_protocol():
    """Test that MockHook implements the MoveHook protocol."""
    hook = MockHook()
    assert isinstance(hook, MoveHook)


def test_notify_start_and_end(reset_hooks):
    """Test that notify functions call registered hooks."""
    import reachy_mini.motion.hooks as hooks_module

    hook = MockHook()
    hooks_module._hooks.append(hook)

    notify_start("happy", "pollen-robotics/reachy-mini-emotions-library")
    assert len(hook.started_moves) == 1
    assert hook.started_moves[0] == ("happy", "pollen-robotics/reachy-mini-emotions-library")

    notify_end("happy", "pollen-robotics/reachy-mini-emotions-library")
    assert len(hook.ended_moves) == 1
    assert hook.ended_moves[0] == ("happy", "pollen-robotics/reachy-mini-emotions-library")


def test_notify_with_none_values(reset_hooks):
    """Test that notifications work with None values."""
    import reachy_mini.motion.hooks as hooks_module

    hook = MockHook()
    hooks_module._hooks.append(hook)

    notify_start(None, None)
    assert hook.started_moves[0] == (None, None)

    notify_end(None, None)
    assert hook.ended_moves[0] == (None, None)


def test_multiple_hooks(reset_hooks):
    """Test that multiple hooks can be registered and called."""
    import reachy_mini.motion.hooks as hooks_module

    hook1 = MockHook()
    hook2 = MockHook()
    hooks_module._hooks.extend([hook1, hook2])

    notify_start("happy", "emotions")
    assert len(hook1.started_moves) == 1
    assert len(hook2.started_moves) == 1


def test_hook_error_handling(reset_hooks):
    """Test that errors in hooks don't crash the application."""
    import reachy_mini.motion.hooks as hooks_module

    # Add both a faulty hook and a working hook
    faulty = FaultyHook()
    working = MockHook()
    hooks_module._hooks.extend([faulty, working])

    # Should not raise, even though faulty hook raises
    notify_start("test", "dataset")
    notify_end("test", "dataset")

    # Working hook should still have been called
    assert len(working.started_moves) == 1
    assert len(working.ended_moves) == 1


def test_load_hooks_with_entry_points(reset_hooks):
    """Test loading hooks from entry points."""
    import reachy_mini.motion.hooks as hooks_module

    # Create a mock entry point
    mock_ep = MagicMock()
    mock_ep.name = "test_hook"
    mock_ep.value = "test_module:TestHook"
    mock_ep.load.return_value = MockHook

    with patch("reachy_mini.motion.hooks.entry_points") as mock_entry_points:
        mock_entry_points.return_value = [mock_ep]
        load_hooks()

    # Should have loaded one hook
    assert len(hooks_module._hooks) == 1
    assert isinstance(hooks_module._hooks[0], MockHook)


def test_load_hooks_only_once(reset_hooks):
    """Test that load_hooks only loads once."""
    import reachy_mini.motion.hooks as hooks_module

    mock_ep = MagicMock()
    mock_ep.name = "test_hook"
    mock_ep.load.return_value = MockHook

    with patch("reachy_mini.motion.hooks.entry_points") as mock_entry_points:
        mock_entry_points.return_value = [mock_ep]
        
        load_hooks()
        initial_count = len(hooks_module._hooks)
        
        load_hooks()  # Should not load again
        assert len(hooks_module._hooks) == initial_count


def test_load_hooks_handles_errors(reset_hooks, caplog):
    """Test that errors during hook loading are handled gracefully."""
    import reachy_mini.motion.hooks as hooks_module

    mock_ep = MagicMock()
    mock_ep.name = "broken_hook"
    mock_ep.load.side_effect = ImportError("Module not found")

    with patch("reachy_mini.motion.hooks.entry_points") as mock_entry_points:
        mock_entry_points.return_value = [mock_ep]
        load_hooks()

    # Should have logged a warning but not crashed
    assert "Failed to load motion hook" in caplog.text


def test_recorded_move_with_dataset_name():
    """Test that RecordedMove stores dataset_name."""
    move_data = {
        "description": "happy",
        "time": [0.0, 1.0],
        "set_target_data": [
            {"head": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], "antennas": [0.0, 0.0], "body_yaw": 0.0},
            {"head": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], "antennas": [0.0, 0.0], "body_yaw": 0.0},
        ],
    }
    
    recorded_move = RecordedMove(
        move_data,
        sound_path=None,
        dataset_name="pollen-robotics/reachy-mini-emotions-library",
    )
    
    assert recorded_move.dataset_name == "pollen-robotics/reachy-mini-emotions-library"
    assert recorded_move.description == "happy"


def test_dummy_move_attributes():
    """Test that DummyMove has the required attributes for hooks."""
    move = DummyMove(
        duration=2.0,
        description="test_move",
        dataset_name="test_dataset",
    )
    
    assert hasattr(move, "description")
    assert hasattr(move, "dataset_name")
    assert move.description == "test_move"
    assert move.dataset_name == "test_dataset"


def test_hook_integration_with_getattr(reset_hooks):
    """Test that hooks work with getattr pattern used in play_move."""
    import reachy_mini.motion.hooks as hooks_module

    hook = MockHook()
    hooks_module._hooks.append(hook)

    # Simulate what play_move does
    move = DummyMove(description="happy", dataset_name="emotions")
    move_name = getattr(move, "description", None)
    dataset = getattr(move, "dataset_name", None)

    notify_start(move_name, dataset)
    notify_end(move_name, dataset)

    assert hook.started_moves[0] == ("happy", "emotions")
    assert hook.ended_moves[0] == ("happy", "emotions")


def test_hook_with_move_without_attributes(reset_hooks):
    """Test that hooks work even if move doesn't have description/dataset_name."""
    import reachy_mini.motion.hooks as hooks_module

    hook = MockHook()
    hooks_module._hooks.append(hook)

    # Move without the optional attributes
    move = DummyMove()
    move_name = getattr(move, "description", None)
    dataset = getattr(move, "dataset_name", None)

    notify_start(move_name, dataset)
    notify_end(move_name, dataset)

    # Should still work, just with None values
    assert hook.started_moves[0] == (None, None)
    assert hook.ended_moves[0] == (None, None)
