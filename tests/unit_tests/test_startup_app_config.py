"""Unit tests for the generic daemon-config helpers.

``_update`` / ``_get_bool`` are the shared primitives behind every per-key
accessor in ``startup_app_config`` (startup app, TURN toggle, first-wake-up
flag). Exercise them directly against a tmp config path (monkeypatched
``_config_path``) so the real user config dir is never touched.
"""

import json
from pathlib import Path

import pytest

from reachy_mini.daemon import startup_app_config


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the module at a throwaway daemon_config.json under a tmp dir."""
    path = tmp_path / "reachy_mini" / "daemon_config.json"
    monkeypatch.setattr(startup_app_config, "_config_path", lambda: path)
    return path


def _stored(config_path: Path) -> dict:
    return json.loads(config_path.read_text())


# ─── _update ─────────────────────────────────────────────────────────────


def test_update_creates_file_and_parent_dirs(config_path: Path):
    assert not config_path.parent.exists()
    startup_app_config._update("some_key", "value")
    assert _stored(config_path) == {"some_key": "value"}


def test_update_overwrites_existing_key(config_path: Path):
    startup_app_config._update("some_key", "old")
    startup_app_config._update("some_key", "new")
    assert _stored(config_path) == {"some_key": "new"}


def test_update_preserves_other_keys(config_path: Path):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('{"startup_app": "face_tracker", "turn_enabled": true}')
    startup_app_config._update("some_key", 42)
    assert _stored(config_path) == {
        "startup_app": "face_tracker",
        "turn_enabled": True,
        "some_key": 42,
    }


def test_update_none_clears_key(config_path: Path):
    startup_app_config._update("some_key", "value")
    startup_app_config._update("some_key", None)
    assert _stored(config_path) == {}


def test_update_none_on_absent_key_writes_empty_config(config_path: Path):
    # Not a no-op on disk: clearing a key a fresh install never had still
    # creates the config file, holding an empty object.
    startup_app_config._update("some_key", None)
    assert _stored(config_path) == {}


def test_update_recovers_from_corrupt_file(config_path: Path):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{ not valid json")
    startup_app_config._update("some_key", "value")
    assert _stored(config_path) == {"some_key": "value"}


def test_update_raises_on_write_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # A plain file where the config's parent dir should be: mkdir and open
    # fail for real, without patching pathlib globally (which would also
    # break the _read() at the top of _update, masking what is under test).
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    monkeypatch.setattr(
        startup_app_config, "_config_path", lambda: blocked / "daemon_config.json"
    )
    with pytest.raises(OSError):
        startup_app_config._update("some_key", "value")


# ─── _get_bool ───────────────────────────────────────────────────────────


def test_get_bool_none_when_file_missing(config_path: Path):
    assert startup_app_config._get_bool("some_flag") is None


def test_get_bool_none_when_key_missing(config_path: Path):
    startup_app_config._update("other_key", True)
    assert startup_app_config._get_bool("some_flag") is None


@pytest.mark.parametrize("value", [True, False])
def test_get_bool_round_trips(config_path: Path, value: bool):
    startup_app_config._update("some_flag", value)
    assert startup_app_config._get_bool("some_flag") is value


@pytest.mark.parametrize("malformed", ["yes", 1, 0, [], {}, None])
def test_get_bool_none_and_warns_on_malformed(
    config_path: Path,
    malformed: object,
    caplog: pytest.LogCaptureFixture,
):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"some_flag": malformed}))
    with caplog.at_level("WARNING"):
        assert startup_app_config._get_bool("some_flag") is None
    assert "some_flag" in caplog.text
