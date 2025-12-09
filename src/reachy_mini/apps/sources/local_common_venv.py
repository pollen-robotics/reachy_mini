"""Utilities for local common venv apps source."""

import logging
import shutil
from importlib.metadata import entry_points

from .. import AppInfo, SourceKind
from ..utils import running_command


async def list_available_apps() -> list[AppInfo]:
    """List apps available from entry points."""
    entry_point_apps = list(entry_points(group="reachy_mini_apps"))
    apps = []

    for ep in entry_point_apps:
        custom_app_url = None
        try:
            app = ep.load()
            custom_app_url = app.custom_app_url
        except Exception as e:
            logging.getLogger("reachy_mini.apps").warning(
                f"Could not load app '{ep.name}' from entry point: {e}"
            )
        apps.append(
            AppInfo(
                name=ep.name,
                source_kind=SourceKind.INSTALLED,
                extra={"custom_app_url": custom_app_url},
            )
        )

    return apps


async def _check_uv_available() -> bool:
    """Check if uv is installed and available.
    
    Returns:
        bool: True if uv is available, False otherwise.

    """
    return shutil.which("uv") is not None


async def _check_git_available() -> bool:
    """Check if git is installed and available.
    
    Returns:
        bool: True if git is available, False otherwise.

    """
    return shutil.which("git") is not None


async def _check_git_lfs() -> bool:
    """Check if git-lfs is installed and configured.
    
    Uses shutil.which() which automatically handles:
    - Windows: looks for git-lfs.exe using PATHEXT
    - Linux/macOS: looks for git-lfs in PATH
    
    Returns:
        bool: True if git-lfs is available, False otherwise.

    """
    if shutil.which("git-lfs") is None:
        return False
    
    # Verify it actually works by running it
    try:
        result = await running_command(
            ["git", "lfs", "version"],
            logger=logging.getLogger("reachy_mini.apps"),
        )
        return result == 0
    except Exception:
        return False


async def install_package(app: AppInfo, logger: logging.Logger) -> int:
    """Install a package given an AppInfo object using uv."""
    if not await _check_uv_available():
        logger.error("uv is not installed. Install with: pip install uv")
        return 1
    
    if app.source_kind == SourceKind.HF_SPACE:
        if not await _check_git_available():
            logger.error("git is not installed")
            return 1
        
        target = f"git+{app.url}" if app.url is not None else app.name
        has_lfs = await _check_git_lfs()
        
        if has_lfs:
            logger.info(f"Adding package {target} using uv with LFS support...")
            return await running_command(["uv", "add", "--lfs", target], logger=logger)
        else:
            logger.warning("git-lfs not found - large files may not download correctly")
            logger.info(f"Installing package {target} using uv pip...")
            return await running_command(["uv", "pip", "install", target], logger=logger)
        
    elif app.source_kind == SourceKind.LOCAL:
        target = app.extra.get("path", app.name)
        logger.info(f"Adding package from local path {target}...")
        return await running_command(["uv", "add", target], logger=logger)
    else:
        raise ValueError(f"Cannot install app from source kind '{app.source_kind}'")


async def uninstall_package(app_name: str, logger: logging.Logger) -> int:
    """Uninstall a package given an app name using uv."""
    # Check if uv is available
    if not await _check_uv_available():
        logger.error(
            "uv is not installed. Install uv to use this package manager.\n"
            "See installation instructions above."
        )
        return 1
    
    existing_apps = await list_available_apps()
    if app_name not in [app.name for app in existing_apps]:
        raise ValueError(f"Cannot uninstall app '{app_name}': it is not installed")

    logger.info(f"Removing package {app_name} using uv...")
    return await running_command(["uv", "remove", app_name], logger=logger)