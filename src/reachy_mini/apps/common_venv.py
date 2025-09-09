"""Used the virtual environment where daemon is installed for management of Reachy Mini apps."""

import sys
from importlib.metadata import entry_points

from .manager import AppInfo, SourceKind
from .utils import running_command


def list_apps_from_entry_points(entry_point: str) -> list[AppInfo]:
    """List apps available from entry points."""
    entry_point_apps = list(entry_points(group=entry_point))
    return [
        AppInfo(name=ep.name, source_kind=SourceKind.INSTALLED)
        for ep in entry_point_apps
    ]


async def install_package(app: AppInfo) -> None:
    """Install a package given an AppInfo object, streaming logs."""
    target = app.url if app.url is not None else app.name

    await running_command(
        [sys.executable, "-m", "pip", "install", target],
        "reachy_mini.apps.manager.install",
    )


async def uninstall_package(app: AppInfo) -> None:
    """Uninstall a package given an AppInfo object."""
    await running_command(
        [sys.executable, "-m", "pip", "uninstall", "-y", app.name],
        "reachy_mini.apps.manager.uninstall",
    )


async def update_package(app: AppInfo) -> None:
    """Update a package given an AppInfo object."""
    target = app.url if app.url is not None else app.name
    await running_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", target],
        "reachy_mini.apps.manager.update",
    )
