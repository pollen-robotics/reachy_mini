"""Unit tests for the persisted first-wake-up flag helpers.

Exercise ``get_first_wake_up_completed`` / ``set_first_wake_up_completed``
against a tmp daemon config path (monkeypatched ``_config_path``) so the real
user config dir is never touched. The helpers are deliberately fail-safe: a
missing file or any read/write error degrades to "not completed" / "write
failed" instead of raising, so a storage problem can't break the command loop.
"""

from pathlib import Path

import pytest

from reachy_mini.daemon import startup_app_config


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the module at a throwaway daemon_config.json under a tmp dir."""
    path = tmp_path / "reachy_mini" / "daemon_config.json"
    monkeypatch.setattr(startup_app_config, "_config_path", lambda: path)
    return path


def test_defaults_to_false_when_unset(config_path: Path):
    assert startup_app_config.get_first_wake_up_completed() is False


def test_set_true_then_get_round_trips(config_path: Path):
    assert startup_app_config.set_first_wake_up_completed(True) is True
    assert startup_app_config.get_first_wake_up_completed() is True


def test_set_false_then_get_round_trips(config_path: Path):
    startup_app_config.set_first_wake_up_completed(True)
    assert startup_app_config.set_first_wake_up_completed(False) is True
    assert startup_app_config.get_first_wake_up_completed() is False


def test_set_creates_parent_dirs(config_path: Path):
    assert not config_path.parent.exists()
    startup_app_config.set_first_wake_up_completed(True)
    assert config_path.is_file()


def test_get_is_failsafe_on_corrupt_json(config_path: Path):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{ not valid json")
    assert startup_app_config.get_first_wake_up_completed() is False


def test_get_defaults_false_when_flag_missing_from_file(config_path: Path):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('{"startup_app": "face_tracker"}')
    assert startup_app_config.get_first_wake_up_completed() is False


def test_get_defaults_false_on_malformed_flag(config_path: Path):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('{"first_wake_up_completed": "yes"}')
    assert startup_app_config.get_first_wake_up_completed() is False


def test_set_preserves_other_config_keys(config_path: Path):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('{"startup_app": "face_tracker"}')
    startup_app_config.set_first_wake_up_completed(True)
    assert startup_app_config.get_startup_app() == "face_tracker"
    assert startup_app_config.get_first_wake_up_completed() is True


def test_set_is_failsafe_on_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # A plain file where the config's parent dir should be, so the write
    # fails for real instead of patching pathlib globally.
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    monkeypatch.setattr(
        startup_app_config, "_config_path", lambda: blocked / "daemon_config.json"
    )
    assert startup_app_config.set_first_wake_up_completed(True) is False
