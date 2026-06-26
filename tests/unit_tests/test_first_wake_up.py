"""Unit tests for the persisted first-wake-up flag helpers.

Exercise ``get_first_wake_up_completed`` / ``set_first_wake_up_completed``
against a tmp config path (monkeypatched ``_STATE_PATH``) so the real
``~/.config`` is never touched. The helpers are deliberately fail-safe: a
missing file or any read/write error degrades to "not completed" / "write
failed" instead of raising, so a storage problem can't break the command loop.
"""

from pathlib import Path

import pytest

from reachy_mini.utils import first_wake_up


@pytest.fixture
def state_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the module at a throwaway JSON file under a tmp config dir."""
    path = tmp_path / ".config" / "reachy_mini" / "first_wake_up.json"
    monkeypatch.setattr(first_wake_up, "_STATE_PATH", path)
    return path


def test_defaults_to_false_when_unset(state_path: Path):
    assert first_wake_up.get_first_wake_up_completed() is False


def test_set_true_then_get_round_trips(state_path: Path):
    assert first_wake_up.set_first_wake_up_completed(True) is True
    assert first_wake_up.get_first_wake_up_completed() is True


def test_set_false_then_get_round_trips(state_path: Path):
    first_wake_up.set_first_wake_up_completed(True)
    assert first_wake_up.set_first_wake_up_completed(False) is True
    assert first_wake_up.get_first_wake_up_completed() is False


def test_set_creates_parent_dirs(state_path: Path):
    assert not state_path.parent.exists()
    first_wake_up.set_first_wake_up_completed(True)
    assert state_path.is_file()


def test_get_is_failsafe_on_corrupt_json(state_path: Path):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{ not valid json")
    assert first_wake_up.get_first_wake_up_completed() is False


def test_get_defaults_false_when_flag_missing_from_file(state_path: Path):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text('{"something_else": true}')
    assert first_wake_up.get_first_wake_up_completed() is False


def test_set_is_failsafe_on_write_error(
    state_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def boom(*a, **k):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(first_wake_up.Path, "write_text", boom)
    assert first_wake_up.set_first_wake_up_completed(True) is False
