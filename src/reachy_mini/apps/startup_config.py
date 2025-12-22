"""Utility functions for managing app startup preferences."""

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_venv_parent_dir() -> Path:
    """Get the parent directory of the current venv (OS-agnostic)."""
    executable = Path(sys.executable)
    
    # Determine expected subdirectory based on platform
    import platform as platform_module
    expected_subdir = "Scripts" if platform_module.system() == "Windows" else "bin"
    
    # Go up from bin/python or Scripts/python.exe to venv dir, then to parent
    if executable.parent.name == expected_subdir:
        venv_dir = executable.parent.parent
        return venv_dir.parent
    
    # Fallback: assume we're already in the venv root
    return executable.parent.parent


def get_startup_config_path() -> Path:
    """Get the path to the startup configuration file."""
    venv_parent_dir = _get_venv_parent_dir()
    config_path = venv_parent_dir / "app_startup_config.json"
    return config_path


def load_startup_config() -> dict[str, bool]:
    """Load the startup configuration from file.
    
    Returns:
        Dictionary mapping app names to their startup preference (True = start at startup).
    """
    config_path = get_startup_config_path()
    
    if not config_path.exists():
        return {}
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            # Validate that all values are booleans
            return {app_name: bool(value) for app_name, value in config.items()}
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load startup config from {config_path}: {e}")
        return {}


def save_startup_config(config: dict[str, bool]) -> None:
    """Save the startup configuration to file.
    
    Args:
        config: Dictionary mapping app names to their startup preference.
    """
    config_path = get_startup_config_path()
    
    try:
        # Ensure parent directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    except IOError as e:
        logger.error(f"Failed to save startup config to {config_path}: {e}")
        raise


def get_app_startup_preference(app_name: str) -> bool:
    """Get the startup preference for a specific app.
    
    Args:
        app_name: Name of the app.
        
    Returns:
        True if the app should start at startup, False otherwise.
    """
    config = load_startup_config()
    return config.get(app_name, False)


def set_app_startup_preference(app_name: str, start_at_startup: bool) -> None:
    """Set the startup preference for a specific app.
    
    Only one app can be set to start at startup at a time. If setting an app to True,
    all other apps will be cleared.
    
    Args:
        app_name: Name of the app.
        start_at_startup: True if the app should start at startup, False otherwise.
        
    Raises:
        ValueError: If multiple apps are already set to start at startup (invalid state).
    """
    config = load_startup_config()
    
    # Validate that at most one app is set to start at startup
    apps_to_start = [name for name, should_start in config.items() if should_start]
    if len(apps_to_start) > 1:
        logger.error(f"Invalid state: multiple apps set to start at startup: {apps_to_start}. Clearing all.")
        # Clear all apps to reset the state
        config = {}
    elif len(apps_to_start) == 1 and start_at_startup and apps_to_start[0] != app_name:
        # Another app is already set to start, clear it
        config.pop(apps_to_start[0], None)
    
    if start_at_startup:
        config[app_name] = True
    else:
        # Remove the entry if set to False (to keep config clean)
        config.pop(app_name, None)
    
    save_startup_config(config)


def get_apps_to_start_at_startup() -> list[str]:
    """Get the list of app names that should start at startup.
    
    Validates that at most one app is set to start. If multiple apps are found,
    clears all and returns an empty list.
    
    Returns:
        List of app names that have start_at_startup set to True (max 1).
    """
    config = load_startup_config()
    apps_to_start = [app_name for app_name, should_start in config.items() if should_start]
    
    # If multiple apps are set, clear all and reset
    if len(apps_to_start) > 1:
        logger.error(f"Invalid state: multiple apps set to start at startup: {apps_to_start}. Clearing all.")
        save_startup_config({})
        return []
    
    return apps_to_start

