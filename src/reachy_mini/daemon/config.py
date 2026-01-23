"""Daemon configuration management.

Handles loading and saving daemon configuration from ~/.reachy_mini/daemon_config.yaml.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".reachy_mini"
CONFIG_FILE = CONFIG_DIR / "daemon_config.yaml"


@dataclass
class AutostartConfig:
    """Configuration for app autostart on daemon boot."""

    enabled: bool = False
    app_name: str | None = None


@dataclass
class DaemonConfig:
    """Root daemon configuration."""

    autostart: AutostartConfig = field(default_factory=AutostartConfig)


def load_daemon_config() -> DaemonConfig:
    """Load daemon configuration from disk.

    Returns defaults if the file is missing or malformed.
    """
    if not CONFIG_FILE.exists():
        logger.debug(f"Config file {CONFIG_FILE} not found, using defaults")
        return DaemonConfig()

    try:
        with open(CONFIG_FILE, "r") as f:
            data = yaml.safe_load(f) or {}

        autostart_data = data.get("autostart", {})
        autostart = AutostartConfig(
            enabled=autostart_data.get("enabled", False),
            app_name=autostart_data.get("app_name"),
        )

        return DaemonConfig(autostart=autostart)

    except yaml.YAMLError as e:
        logger.warning(f"Malformed config file {CONFIG_FILE}: {e}, using defaults")
        return DaemonConfig()
    except Exception as e:
        logger.warning(f"Error reading config file {CONFIG_FILE}: {e}, using defaults")
        return DaemonConfig()


def save_daemon_config(config: DaemonConfig) -> None:
    """Save daemon configuration to disk.

    Creates the config directory if it doesn't exist.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "autostart": {
            "enabled": config.autostart.enabled,
            "app_name": config.autostart.app_name,
        }
    }

    with open(CONFIG_FILE, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)

    logger.info(f"Saved daemon config to {CONFIG_FILE}")
