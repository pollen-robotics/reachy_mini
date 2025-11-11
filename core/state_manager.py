"""Session state manager for Reachy Control Center.

Handles persistence of UI state, preferences, and extension configurations.
Inspired by LAURA's chat_log persistence pattern.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class StateManager:
    """Manages persistent session state for control center."""

    def __init__(self, state_file: str = "control_center_state.json"):
        """Initialize state manager.

        Args:
            state_file: Path to state file (relative to cwd)
        """
        self.state_file = Path(state_file)
        self.state: Dict[str, Any] = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load state from disk.

        Returns:
            State dictionary or empty default state
        """
        if not self.state_file.exists():
            logger.info("No previous state found, starting fresh")
            return self._default_state()

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                logger.info(f"Loaded state from {self.state_file}")
                return state
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return self._default_state()

    def _default_state(self) -> Dict[str, Any]:
        """Create default state dictionary.

        Returns:
            Default state with all settings at defaults
        """
        return {
            "version": "1.0.0",
            "last_updated": datetime.now().isoformat(),
            "daemon": {
                "url": "http://localhost:8100",
                "auto_start": False,
                "auto_stop": False
            },
            "manual_control": {
                "head_x": 0.0,
                "head_y": 0.0,
                "head_z": 0.0,
                "head_yaw": 0.0,
                "head_pitch": 0.0,
                "head_roll": 0.0,
                "left_antenna": 0.0,
                "right_antenna": 0.0,
                "duration": 0.5
            },
            "extensions": {
                "running": [],  # List of extension names that were running
                "configs": {}   # Per-extension configuration
            },
            "ui": {
                "last_tab": "Manual Control",
                "window_size": [1920, 1080]
            }
        }

    def save_state(self) -> bool:
        """Save current state to disk.

        Returns:
            True if save succeeded, False otherwise
        """
        try:
            self.state["last_updated"] = datetime.now().isoformat()

            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)

            logger.debug(f"Saved state to {self.state_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from state.

        Args:
            key: Dot-notation key (e.g., "daemon.url")
            default: Default value if key not found

        Returns:
            Value or default
        """
        keys = key.split('.')
        value = self.state

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any, auto_save: bool = True) -> None:
        """Set value in state.

        Args:
            key: Dot-notation key (e.g., "daemon.url")
            value: Value to set
            auto_save: Automatically save to disk after setting
        """
        keys = key.split('.')
        current = self.state

        # Navigate to the parent dict
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Set the value
        current[keys[-1]] = value

        if auto_save:
            self.save_state()

    def get_manual_control_state(self) -> Dict[str, float]:
        """Get all manual control slider values.

        Returns:
            Dictionary of slider values
        """
        return self.get("manual_control", {})

    def set_manual_control_state(self, **kwargs) -> None:
        """Set manual control slider values.

        Args:
            **kwargs: Slider values (head_x, head_y, etc.)
        """
        for key, value in kwargs.items():
            self.set(f"manual_control.{key}", value, auto_save=False)
        self.save_state()

    def get_daemon_config(self) -> Dict[str, Any]:
        """Get daemon configuration.

        Returns:
            Daemon config dictionary
        """
        return self.get("daemon", {})

    def set_daemon_url(self, url: str) -> None:
        """Set daemon URL.

        Args:
            url: Daemon URL
        """
        self.set("daemon.url", url)

    def get_running_extensions(self) -> list:
        """Get list of extensions that were running.

        Returns:
            List of extension names
        """
        return self.get("extensions.running", [])

    def set_running_extensions(self, extensions: list) -> None:
        """Set list of running extensions.

        Args:
            extensions: List of extension names
        """
        self.set("extensions.running", extensions)

    def get_extension_config(self, extension_name: str) -> Dict[str, Any]:
        """Get configuration for specific extension.

        Args:
            extension_name: Name of extension

        Returns:
            Extension config dictionary
        """
        return self.get(f"extensions.configs.{extension_name}", {})

    def set_extension_config(self, extension_name: str, config: Dict[str, Any]) -> None:
        """Set configuration for specific extension.

        Args:
            extension_name: Name of extension
            config: Configuration dictionary
        """
        self.set(f"extensions.configs.{extension_name}", config)

    def cleanup(self) -> None:
        """Cleanup and final save before shutdown."""
        logger.info("Saving final state before shutdown")
        self.save_state()
