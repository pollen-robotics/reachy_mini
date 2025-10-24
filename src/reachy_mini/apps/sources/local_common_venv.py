"""Utilities for local common venv apps source."""

import logging
import sys
from importlib.metadata import entry_points

from .. import AppInfo, SourceKind
from ..utils import running_command


async def list_available_apps() -> list[AppInfo]:
    """List apps available from entry points."""
    entry_point_apps = list(entry_points(group="reachy_mini_apps"))
    return [
        AppInfo(name=ep.name, source_kind=SourceKind.INSTALLED)
        for ep in entry_point_apps
    ]


async def install_package(app: AppInfo, logger: logging.Logger) -> int:
    """Install a package given an AppInfo object, streaming logs."""
    if app.source_kind == SourceKind.HF_SPACE:
        target = f"git+{app.url}" if app.url is not None else app.name
    elif app.source_kind == SourceKind.LOCAL:
        target = app.extra.get("path", app.name)
    else:
        raise ValueError(f"Cannot install app from source kind '{app.source_kind}'")

    return await running_command(
        [sys.executable, "-m", "pip", "install", target],
        logger=logger,
    )


async def uninstall_package(app_name: str, logger: logging.Logger) -> int:
    """Uninstall a package given an app name."""
    existing_apps = await list_available_apps()
    if app_name not in [app.name for app in existing_apps]:
        raise ValueError(f"Cannot uninstall app '{app_name}': it is not installed")

    return await running_command(
        [sys.executable, "-m", "pip", "uninstall", "-y", app_name],
        logger=logger,
    )
