# apps/dashboard/__init__.py
"""
Robot Dashboard Package

This package provides a web-based dashboard for managing robot applications.
"""

__version__ = "0.1.0"

# Import main components for easier access
from .dashboard_api import app, start_app_by_name, stop_app
from .utils import IS_LINUX, IS_MACOS, IS_WINDOWS, get_platform_info
from .venv_app import VenvAppManager

__all__ = [
    "app",
    "start_app_by_name",
    "stop_app",
    "VenvAppManager",
    "get_platform_info",
    "IS_WINDOWS",
    "IS_MACOS",
    "IS_LINUX",
]
