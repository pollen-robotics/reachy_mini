"""Reachy Mini SDK."""

from reachy_mini.apps.app import ReachyMiniApp  # noqa: F401
from reachy_mini.reachy_mini import ReachyMini  # noqa: F401

from importlib.metadata import version

__version__ = version("reachy_mini")
