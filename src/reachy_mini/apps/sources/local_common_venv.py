"""Utilities for local common venv apps source."""

import asyncio
import logging
import platform
import shutil
import sys
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, cast

from huggingface_hub import snapshot_download

from .. import AppInfo, SourceKind
from ..utils import running_command

if TYPE_CHECKING:
    from ..app import ReachyMiniApp


def _is_windows() -> bool:
    """Check if the current platform is Windows."""
    return platform.system() == "Windows"


def _should_use_separate_venvs(
    wireless_version: bool = False, desktop_app_daemon: bool = False
) -> bool:
    """Determine if we should use separate venvs based on version flags."""
    # Disable venv for wireless version due to storage constraints
    return desktop_app_daemon


def _get_venv_parent_dir() -> Path:
    """Get the parent directory of the current venv (OS-agnostic)."""
    # sys.executable is typically: /path/to/venv/bin/python (Linux/Mac)
    # or: C:\path\to\venv\Scripts\python.exe (Windows)
    executable = Path(sys.executable)

    # Determine expected subdirectory based on platform
    expected_subdir = "Scripts" if _is_windows() else "bin"

    # Go up from bin/python or Scripts/python.exe to venv dir, then to parent
    if executable.parent.name == expected_subdir:
        venv_dir = executable.parent.parent
        return venv_dir.parent

    # Fallback: assume we're already in the venv root
    return executable.parent.parent


def _get_app_venv_path(app_name: str) -> Path:
    """Get the venv path for a given app (sibling to current venv)."""
    parent_dir = _get_venv_parent_dir()
    return parent_dir / f"{app_name}_venv"


def _get_app_python(app_name: str) -> Path:
    """Get the Python executable path for a given app (OS-agnostic)."""
    venv_path = _get_app_venv_path(app_name)

    if _is_windows():
        # Windows: Scripts/python.exe
        python_exe = venv_path / "Scripts" / "python.exe"
        if python_exe.exists():
            return python_exe
        # Fallback without .exe
        python_path = venv_path / "Scripts" / "python"
        if python_path.exists():
            return python_path
        # Default
        return venv_path / "Scripts" / "python.exe"
    else:
        # Linux/Mac: bin/python
        python_path = venv_path / "bin" / "python"
        if python_path.exists():
            return python_path
        # Default
        return venv_path / "bin" / "python"


def _get_app_site_packages(app_name: str) -> Path | None:
    """Get the site-packages directory for a given app's venv (OS-agnostic)."""
    venv_path = _get_app_venv_path(app_name)

    if _is_windows():
        # Windows: Lib/site-packages
        site_packages = venv_path / "Lib" / "site-packages"
        if site_packages.exists():
            return site_packages
        return None
    else:
        # Linux/Mac: lib/python3.x/site-packages
        lib_dir = venv_path / "lib"
        if not lib_dir.exists():
            return None
        python_dirs = list(lib_dir.glob("python3.*"))
        if not python_dirs:
            return None
        return python_dirs[0] / "site-packages"


def get_app_site_packages(app_name: str) -> Path | None:
    """Public API to get the site-packages directory for a given app's venv."""
    return _get_app_site_packages(app_name)


async def _list_apps_from_separate_venvs() -> list[AppInfo]:
    """List apps by scanning sibling venv directories."""
    parent_dir = _get_venv_parent_dir()
    if not parent_dir.exists():
        return []

    apps = []
    for venv_path in parent_dir.iterdir():
        if not venv_path.is_dir() or not venv_path.name.endswith("_venv"):
            continue

        # Extract app name from venv directory name
        app_name = venv_path.name[: -len("_venv")]

        # Note: We don't load the app to get custom_app_url for separate venvs
        # to avoid sys.path pollution and version conflicts. The custom_app_url
        # will still work when the app actually runs (it uses its own class attribute).
        # This only means the settings icon won't appear in the dashboard listing.
        custom_app_url = None

        apps.append(
            AppInfo(
                name=app_name,
                source_kind=SourceKind.INSTALLED,
                extra={
                    "custom_app_url": custom_app_url,
                    "venv_path": str(venv_path),
                },
            )
        )

    return apps


async def _list_apps_from_entry_points() -> list[AppInfo]:
    """List apps from current environment's entry points."""
    entry_point_apps = entry_points(group="reachy_mini_apps")

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


async def list_available_apps(
    wireless_version: bool = False, desktop_app_daemon: bool = False
) -> list[AppInfo]:
    """List apps available from entry points or separate venvs."""
    if _should_use_separate_venvs(wireless_version, desktop_app_daemon):
        return await _list_apps_from_separate_venvs()
    else:
        return await _list_apps_from_entry_points()


async def install_package(
    app: AppInfo,
    logger: logging.Logger,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> int:
    """Install a package given an AppInfo object, streaming logs."""
    if app.source_kind == SourceKind.HF_SPACE:
        # Use huggingface_hub to download the repo (handles LFS automatically)
        # This avoids requiring git-lfs to be installed on the system
        if app.url is not None:
            # Extract repo_id from URL like "https://huggingface.co/spaces/owner/repo"
            parts = app.url.rstrip("/").split("/")
            repo_id = f"{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else app.name
        else:
            repo_id = app.name

        logger.info(f"Downloading HuggingFace Space: {repo_id}")
        try:
            target = await asyncio.to_thread(
                snapshot_download,
                repo_id=repo_id,
                repo_type="space",
            )
            logger.info(f"Downloaded to: {target}")
        except Exception as e:
            logger.error(f"Failed to download from HuggingFace: {e}")
            return 1
    elif app.source_kind == SourceKind.LOCAL:
        target = app.extra.get("path", app.name)
    else:
        raise ValueError(f"Cannot install app from source kind '{app.source_kind}'")

    if _should_use_separate_venvs(wireless_version, desktop_app_daemon):
        # Create separate venv for this app
        app_name = app.name
        venv_path = _get_app_venv_path(app_name)
        success = False

        # Remove existing venv if it exists
        if venv_path.exists():
            logger.info(f"Removing existing venv at {venv_path}")
            shutil.rmtree(venv_path)

        try:
            # Create venv using python -m venv
            logger.info(f"Creating venv for '{app_name}' at {venv_path}")
            ret = await running_command(
                [sys.executable, "-m", "venv", str(venv_path)], logger=logger
            )
            if ret != 0:
                return ret

            # Install package in the new venv
            python_path = _get_app_python(app_name)
            ret = await running_command(
                [str(python_path), "-m", "pip", "install", target],
                logger=logger,
            )

            if ret != 0:
                return ret

            logger.info(f"Successfully installed '{app_name}' in {venv_path}")
            success = True
            return 0
        finally:
            # Clean up broken venv on any failure
            if not success and venv_path.exists():
                logger.warning(f"Installation failed, cleaning up {venv_path}")
                shutil.rmtree(venv_path)
    else:
        # Original behavior: install into current environment
        return await running_command(
            [sys.executable, "-m", "pip", "install", target],
            logger=logger,
        )


def load_app_from_venv(
    app_name: str,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> type["ReachyMiniApp"]:
    """Load an app class from its separate venv or current environment."""
    if _should_use_separate_venvs(wireless_version, desktop_app_daemon):
        # Load from separate venv
        site_packages = _get_app_site_packages(app_name)
        if not site_packages or not site_packages.exists():
            raise ValueError(f"App '{app_name}' venv not found or invalid")

        sys.path.insert(0, str(site_packages))
        try:
            eps = entry_points(group="reachy_mini_apps")
            ep = eps.select(name=app_name)
            if not ep:
                raise ValueError(f"No entry point found for app '{app_name}'")
            app_cls = list(ep)[0].load()
            return cast(type["ReachyMiniApp"], app_cls)
        except Exception as e:
            raise ValueError(f"Could not load app '{app_name}' from venv: {e}")
        finally:
            sys.path.pop(0)
    else:
        # Original behavior: load from current environment
        eps = entry_points(group="reachy_mini_apps", name=app_name)
        ep_list = list(eps)
        if not ep_list:
            raise ValueError(f"No entry point found for app '{app_name}'")
        return cast(type["ReachyMiniApp"], ep_list[0].load())


async def uninstall_package(
    app_name: str,
    logger: logging.Logger,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> int:
    """Uninstall a package given an app name."""
    if _should_use_separate_venvs(wireless_version, desktop_app_daemon):
        # Remove the venv directory
        venv_path = _get_app_venv_path(app_name)

        if not venv_path.exists():
            raise ValueError(f"Cannot uninstall app '{app_name}': it is not installed")

        logger.info(f"Removing venv for '{app_name}' at {venv_path}")
        shutil.rmtree(venv_path)

        logger.info(f"Successfully uninstalled '{app_name}'")
        return 0
    else:
        # Original behavior: uninstall from current environment
        existing_apps = await list_available_apps()
        if app_name not in [app.name for app in existing_apps]:
            raise ValueError(f"Cannot uninstall app '{app_name}': it is not installed")

        return await running_command(
            [sys.executable, "-m", "pip", "uninstall", "-y", app_name],
            logger=logger,
        )
