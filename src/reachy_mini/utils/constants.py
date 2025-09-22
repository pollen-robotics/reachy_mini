"""Utility constants for the reachy_mini package."""

from importlib.resources import files

import reachy_mini

URDF_ROOT_PATH: str = str(files(reachy_mini).joinpath("descriptions/reachy_mini/urdf"))
ASSETS_ROOT_PATH: str = str(files(reachy_mini).joinpath("assets/"))
MODELS_ROOT_PATH: str = str(files(reachy_mini).joinpath("assets/models"))
