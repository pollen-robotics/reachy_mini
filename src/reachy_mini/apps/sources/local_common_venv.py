"""Utilities for local common venv apps source."""

import logging
import shutil
import sys
from importlib.metadata import entry_points
from pathlib import Path

from .. import AppInfo, SourceKind
from ..utils import running_command


def _should_use_separate_venvs(
    wireless_version: bool = False, desktop_version: bool = False
) -> bool:
    """Determine if we should use separate venvs based on version flags."""
    return wireless_version or desktop_version


def _get_venv_parent_dir() -> Path:
    """Get the parent directory of the current venv (OS-agnostic)."""
    # sys.executable is typically: /path/to/venv/bin/python (Linux/Mac)
    # or: C:\path\to\venv\Scripts\python.exe (Windows)
    executable = Path(sys.executable)

    # Go up from bin/python or Scripts/python.exe to venv dir, then to parent
    if executable.parent.name in ("bin", "Scripts"):
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
    # Check for both Linux/Mac (bin) and Windows (Scripts)
    for subdir in ("bin", "Scripts"):
        python_path = venv_path / subdir / "python"
        if python_path.exists():
            return python_path
        # Try with .exe extension for Windows
        python_exe = venv_path / subdir / "python.exe"
        if python_exe.exists():
            return python_exe
    # Default to bin/python
    return venv_path / "bin" / "python"


def _get_app_site_packages(app_name: str) -> Path | None:
    """Get the site-packages directory for a given app's venv (OS-agnostic)."""
    venv_path = _get_app_venv_path(app_name)

    # Check for both Lib/site-packages (Windows) and lib/python3.x/site-packages (Linux/Mac)
    site_packages_win = venv_path / "Lib" / "site-packages"
    if site_packages_win.exists():
        return site_packages_win

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


async def list_available_apps(
    wireless_version: bool = False, desktop_version: bool = False
) -> list[AppInfo]:
    """List apps available from entry points or separate venvs."""
    if _should_use_separate_venvs(wireless_version, desktop_version):
        # List by scanning sibling venv directories
        parent_dir = _get_venv_parent_dir()
        if not parent_dir.exists():
            return []

        apps = []
        for venv_path in parent_dir.iterdir():
            if not venv_path.is_dir() or not venv_path.name.endswith("_venv"):
                continue

            # Extract app name from venv directory name
            app_name = venv_path.name[: -len("_venv")]
            custom_app_url = None

            # Try to load app to get custom_app_url
            site_packages = _get_app_site_packages(app_name)
            if site_packages and site_packages.exists():
                sys.path.insert(0, str(site_packages))
                try:
                    eps = entry_points(group="reachy_mini_apps")
                    ep = eps.select(name=app_name)
                    if ep:
                        app_cls = list(ep)[0].load()
                        custom_app_url = getattr(app_cls, "custom_app_url", None)
                except Exception as e:
                    logging.getLogger("reachy_mini.apps").warning(
                        f"Could not load app '{app_name}': {e}"
                    )
                finally:
                    sys.path.pop(0)

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
    else:
        # Original behavior: list from current environment's entry points
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


async def install_package(
    app: AppInfo,
    logger: logging.Logger,
    wireless_version: bool = False,
    desktop_version: bool = False,
) -> int:
    """Install a package given an AppInfo object, streaming logs."""
    if app.source_kind == SourceKind.HF_SPACE:
        target = f"git+{app.url}" if app.url is not None else app.name
    elif app.source_kind == SourceKind.LOCAL:
        target = app.extra.get("path", app.name)
    else:
        raise ValueError(f"Cannot install app from source kind '{app.source_kind}'")

    if _should_use_separate_venvs(wireless_version, desktop_version):
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


async def uninstall_package(
    app_name: str,
    logger: logging.Logger,
    wireless_version: bool = False,
    desktop_version: bool = False,
) -> int:
    """Uninstall a package given an app name."""
    if _should_use_separate_venvs(wireless_version, desktop_version):
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
