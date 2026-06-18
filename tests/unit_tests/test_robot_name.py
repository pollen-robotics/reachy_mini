"""Unit tests for the persisted robot-name helpers.

Exercise ``get_robot_name`` / ``set_robot_name`` against a tmp config path
(monkeypatched ``_STATE_PATH``) so the real ``~/.config`` is never touched.
The helpers are deliberately fail-safe: every error path returns ``None``
instead of raising, so a storage problem can't break the daemon command loop.
"""

from pathlib import Path

import pytest

from reachy_mini.utils import robot_name


@pytest.fixture
def state_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the module at a throwaway JSON file under a tmp config dir."""
    path = tmp_path / ".config" / "reachy_mini" / "robot_name.json"
    monkeypatch.setattr(robot_name, "_STATE_PATH", path)
    return path


def test_get_returns_none_when_unset(state_path: Path):
    assert robot_name.get_robot_name() is None


def test_set_then_get_round_trips(state_path: Path):
    assert robot_name.set_robot_name("Wall-E") == "Wall-E"
    assert robot_name.get_robot_name() == "Wall-E"


def test_set_creates_parent_dirs(state_path: Path):
    assert not state_path.parent.exists()
    robot_name.set_robot_name("R2D2")
    assert state_path.is_file()


def test_set_trims_surrounding_whitespace(state_path: Path):
    assert robot_name.set_robot_name("  Eve  ") == "Eve"
    assert robot_name.get_robot_name() == "Eve"


def test_set_caps_length_to_max(state_path: Path):
    long_name = "x" * 200
    stored = robot_name.set_robot_name(long_name)
    assert stored == "x" * robot_name._MAX_NAME_LENGTH
    assert robot_name.get_robot_name() == "x" * robot_name._MAX_NAME_LENGTH


@pytest.mark.parametrize("value", ["", "   ", "\n\t"])
def test_set_rejects_empty_or_whitespace(state_path: Path, value: str):
    assert robot_name.set_robot_name(value) is None
    assert not state_path.exists()


def test_set_rejects_non_string(state_path: Path):
    assert robot_name.set_robot_name(123) is None  # type: ignore[arg-type]
    assert not state_path.exists()


def test_get_is_failsafe_on_corrupt_json(state_path: Path):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{ not valid json")
    assert robot_name.get_robot_name() is None


def test_get_ignores_non_string_name(state_path: Path):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text('{"name": 42}')
    assert robot_name.get_robot_name() is None


def test_set_is_failsafe_on_write_error(
    state_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def boom(*a, **k):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(robot_name.Path, "write_text", boom)
    assert robot_name.set_robot_name("Hal") is None
