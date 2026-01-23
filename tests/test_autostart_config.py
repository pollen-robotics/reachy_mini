"""Tests for daemon autostart configuration."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from reachy_mini.daemon.config import (
    CONFIG_FILE,
    AutostartConfig,
    DaemonConfig,
    load_daemon_config,
    save_daemon_config,
)


@pytest.fixture
def temp_config_dir(tmp_path: Path):
    """Temporarily redirect config file to a temp directory."""
    temp_config = tmp_path / "daemon_config.yaml"
    with patch("reachy_mini.daemon.config.CONFIG_FILE", temp_config):
        with patch("reachy_mini.daemon.config.CONFIG_DIR", tmp_path):
            yield tmp_path, temp_config


class TestAutostartConfig:
    """Tests for AutostartConfig dataclass."""

    def test_defaults(self):
        config = AutostartConfig()
        assert config.enabled is False
        assert config.app_name is None

    def test_with_values(self):
        config = AutostartConfig(enabled=True, app_name="test-app")
        assert config.enabled is True
        assert config.app_name == "test-app"


class TestDaemonConfig:
    """Tests for DaemonConfig dataclass."""

    def test_defaults(self):
        config = DaemonConfig()
        assert isinstance(config.autostart, AutostartConfig)
        assert config.autostart.enabled is False


class TestLoadDaemonConfig:
    """Tests for load_daemon_config function."""

    def test_missing_file_returns_defaults(self, temp_config_dir):
        """Config file missing should return defaults."""
        _, temp_config = temp_config_dir
        assert not temp_config.exists()

        config = load_daemon_config()
        assert config.autostart.enabled is False
        assert config.autostart.app_name is None

    def test_valid_config_file(self, temp_config_dir):
        """Valid config file should be loaded correctly."""
        tmp_path, temp_config = temp_config_dir

        data = {"autostart": {"enabled": True, "app_name": "my-app"}}
        with open(temp_config, "w") as f:
            yaml.safe_dump(data, f)

        config = load_daemon_config()
        assert config.autostart.enabled is True
        assert config.autostart.app_name == "my-app"

    def test_malformed_yaml_returns_defaults(self, temp_config_dir):
        """Malformed YAML should return defaults with warning."""
        _, temp_config = temp_config_dir

        with open(temp_config, "w") as f:
            f.write("invalid: yaml: content: [")

        config = load_daemon_config()
        assert config.autostart.enabled is False
        assert config.autostart.app_name is None

    def test_partial_config_uses_defaults(self, temp_config_dir):
        """Missing fields should use defaults."""
        _, temp_config = temp_config_dir

        data = {"autostart": {"enabled": True}}  # Missing app_name
        with open(temp_config, "w") as f:
            yaml.safe_dump(data, f)

        config = load_daemon_config()
        assert config.autostart.enabled is True
        assert config.autostart.app_name is None

    def test_empty_file_returns_defaults(self, temp_config_dir):
        """Empty file should return defaults."""
        _, temp_config = temp_config_dir

        temp_config.touch()

        config = load_daemon_config()
        assert config.autostart.enabled is False


class TestSaveDaemonConfig:
    """Tests for save_daemon_config function."""

    def test_save_creates_file(self, temp_config_dir):
        """Save should create file if it doesn't exist."""
        _, temp_config = temp_config_dir
        assert not temp_config.exists()

        config = DaemonConfig(
            autostart=AutostartConfig(enabled=True, app_name="test-app")
        )
        save_daemon_config(config)

        assert temp_config.exists()

        with open(temp_config, "r") as f:
            data = yaml.safe_load(f)

        assert data["autostart"]["enabled"] is True
        assert data["autostart"]["app_name"] == "test-app"

    def test_save_overwrites_existing(self, temp_config_dir):
        """Save should overwrite existing config."""
        _, temp_config = temp_config_dir

        # Write initial config
        with open(temp_config, "w") as f:
            yaml.safe_dump({"autostart": {"enabled": False, "app_name": "old-app"}}, f)

        # Save new config
        config = DaemonConfig(
            autostart=AutostartConfig(enabled=True, app_name="new-app")
        )
        save_daemon_config(config)

        with open(temp_config, "r") as f:
            data = yaml.safe_load(f)

        assert data["autostart"]["enabled"] is True
        assert data["autostart"]["app_name"] == "new-app"

    def test_save_disabled_config(self, temp_config_dir):
        """Save disabled config should clear app_name."""
        _, temp_config = temp_config_dir

        config = DaemonConfig(autostart=AutostartConfig(enabled=False, app_name=None))
        save_daemon_config(config)

        with open(temp_config, "r") as f:
            data = yaml.safe_load(f)

        assert data["autostart"]["enabled"] is False
        assert data["autostart"]["app_name"] is None


class TestRoundTrip:
    """Tests for save/load roundtrip."""

    def test_roundtrip_enabled(self, temp_config_dir):
        """Config should survive save/load roundtrip."""
        original = DaemonConfig(
            autostart=AutostartConfig(enabled=True, app_name="roundtrip-app")
        )

        save_daemon_config(original)
        loaded = load_daemon_config()

        assert loaded.autostart.enabled == original.autostart.enabled
        assert loaded.autostart.app_name == original.autostart.app_name

    def test_roundtrip_disabled(self, temp_config_dir):
        """Disabled config should survive roundtrip."""
        original = DaemonConfig(autostart=AutostartConfig(enabled=False, app_name=None))

        save_daemon_config(original)
        loaded = load_daemon_config()

        assert loaded.autostart.enabled is False
        assert loaded.autostart.app_name is None
