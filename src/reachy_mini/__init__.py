"""Reachy Mini SDK."""

from importlib.metadata import version

from reachy_mini.apps.app import ReachyMiniApp
from reachy_mini.reachy_mini import ReachyMini

__version__ = version("reachy_mini")

__all__ = ["ReachyMini", "ReachyMiniApp", "__version__"]
