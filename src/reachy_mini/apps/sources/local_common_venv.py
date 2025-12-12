"""Utilities for local common venv apps source."""

import asyncio
import logging
import platform
import re
import shutil
import sys
from importlib.metadata import entry_points
from pathlib import Path

from huggingface_hub import snapshot_download

from .. import AppInfo, SourceKind
from ..utils import running_command


async def _check_uv_available() -> bool:
    """Check if uv is available on the system."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "uv", "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.wait()
        return proc.returncode == 0
    except FileNotFoundError:
        return False


def _is_windows() -> bool:
    """Check if the current platform is Windows."""
    return platform.system() == "Windows"


def _should_use_separate_venvs(
    wireless_version: bool = False, desktop_app_daemon: bool = False
) -> bool:
    """Determine if we should use separate venvs based on version flags."""
    # Use separate venvs for desktop (one per app) and wireless (shared apps venv)
    return desktop_app_daemon or wireless_version


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


def _get_app_venv_path(
    app_name: str,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> Path:
    """Get the venv path for a given app (sibling to current venv).

    On wireless: returns a shared 'apps_venv' for all apps
    On desktop: returns a separate venv per app '{app_name}_venv'
    """
    parent_dir = _get_venv_parent_dir()
    if wireless_version and not desktop_app_daemon:
        # Wireless: shared venv for all apps
        return parent_dir / "apps_venv"
    else:
        # Desktop: separate venv per app
        return parent_dir / f"{app_name}_venv"


def _get_app_python(
    app_name: str,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> Path:
    """Get the Python executable path for a given app (OS-agnostic)."""
    venv_path = _get_app_venv_path(app_name, wireless_version, desktop_app_daemon)

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


def _get_app_site_packages(
    app_name: str,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> Path | None:
    """Get the site-packages directory for a given app's venv (OS-agnostic)."""
    venv_path = _get_app_venv_path(app_name, wireless_version, desktop_app_daemon)

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


def get_app_site_packages(
    app_name: str,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> Path | None:
    """Public API to get the site-packages directory for a given app's venv."""
    return _get_app_site_packages(app_name, wireless_version, desktop_app_daemon)


def get_app_python(
    app_name: str,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> Path:
    """Get the Python executable path for an app (cross-platform).

    For separate venvs: returns the app's venv Python
    For shared environment: returns the current Python interpreter
    """
    if _should_use_separate_venvs(wireless_version, desktop_app_daemon):
        return _get_app_python(app_name, wireless_version, desktop_app_daemon)
    else:
        return Path(sys.executable)


def _get_custom_app_url_from_file(
    app_name: str,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> str | None:
    """Get custom_app_url by reading it from the app's main.py file.

    This is much faster than subprocess and avoids sys.path pollution.
    Looks for patterns like: custom_app_url: str | None = "http://..."
    """
    site_packages = _get_app_site_packages(
        app_name, wireless_version, desktop_app_daemon
    )
    if not site_packages or not site_packages.exists():
        return None

    # Try to find main.py in the app's package directory
    app_dir = site_packages / app_name
    main_file = app_dir / "main.py"

    if not main_file.exists():
        return None

    try:
        content = main_file.read_text(encoding="utf-8")

        # Match patterns like:
        # custom_app_url: str | None = "http://..."
        # custom_app_url = "http://..."
        # custom_app_url: str = "http://..."
        pattern = r'custom_app_url\s*(?::\s*[^=]+)?\s*=\s*["\']([^"\']+)["\']'
        match = re.search(pattern, content)

        if match:
            return match.group(1)
        return None
    except Exception as e:
        logging.getLogger("reachy_mini.apps").warning(
            f"Could not read custom_app_url from '{app_name}/main.py': {e}"
        )
        return None


async def _list_apps_from_separate_venvs(
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> list[AppInfo]:
    """List apps by scanning sibling venv directories or shared venv entry points."""
    parent_dir = _get_venv_parent_dir()
    if not parent_dir.exists():
        return []

    if wireless_version and not desktop_app_daemon:
        # Wireless: list apps from shared venv's entry points using subprocess
        apps_venv = parent_dir / "apps_venv"
        if not apps_venv.exists():
            return []

        # Get Python executable from the apps_venv
        python_path = _get_app_python("dummy", wireless_version, desktop_app_daemon)
        if not python_path.exists():
            return []

        # Use subprocess to list entry points from the apps_venv environment
        import subprocess

        try:
            result = subprocess.run(
                [
                    str(python_path),
                    "-c",
                    "from importlib.metadata import entry_points; "
                    "eps = entry_points(group='reachy_mini_apps'); "
                    "print('\\n'.join(ep.name for ep in eps))",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return []

            app_names = [
                name.strip()
                for name in result.stdout.strip().split("\n")
                if name.strip()
            ]
            apps = []
            for app_name in app_names:
                custom_app_url = _get_custom_app_url_from_file(
                    app_name, wireless_version, desktop_app_daemon
                )
                apps.append(
                    AppInfo(
                        name=app_name,
                        source_kind=SourceKind.INSTALLED,
                        extra={
                            "custom_app_url": custom_app_url,
                            "venv_path": str(apps_venv),
                        },
                    )
                )
            return apps
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []
    else:
        # Desktop: scan for per-app venv directories
        apps = []
        for venv_path in parent_dir.iterdir():
            if not venv_path.is_dir() or not venv_path.name.endswith("_venv"):
                continue

            # Skip the shared wireless apps_venv directory
            if venv_path.name == "apps_venv":
                continue

            # Extract app name from venv directory name
            app_name = venv_path.name[: -len("_venv")]

            # Get custom_app_url by reading the main.py file (fast, no sys.path pollution)
            # This ensures the settings icon appears in the dashboard listing
            custom_app_url = _get_custom_app_url_from_file(
                app_name, wireless_version, desktop_app_daemon
            )

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
        return await _list_apps_from_separate_venvs(
            wireless_version, desktop_app_daemon
        )
    else:
        return await _list_apps_from_entry_points()


async def install_package(
    app: AppInfo,
    logger: logging.Logger,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> int:
    """Install a package given an AppInfo object, streaming logs."""
    # Check if uv is available
    use_uv = await _check_uv_available()
    if not use_uv:
        logger.warning(
            "uv is not installed. Falling back to pip. "
            "Install uv for faster installs: pip install uv"
        )
    
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
        venv_path = _get_app_venv_path(app_name, wireless_version, desktop_app_daemon)
        success = False

        # On wireless, only create venv if it doesn't exist (shared across apps)
        venv_exists = venv_path.exists()
        if venv_exists and not (wireless_version and not desktop_app_daemon):
            # Desktop: remove existing per-app venv
            logger.info(f"Removing existing venv at {venv_path}")
            shutil.rmtree(venv_path)
            venv_exists = False

        try:
            # Create venv if needed
            if not venv_exists:
                logger.info(f"Creating venv for '{app_name}' at {venv_path}")
                ret = await running_command(
                    [sys.executable, "-m", "venv", str(venv_path)], logger=logger
                )
                if ret != 0:
                    return ret

                # On wireless, pre-install reachy-mini with gstreamer support
                if wireless_version and not desktop_app_daemon:
                    logger.info(
                        "Pre-installing reachy-mini with gstreamer support in apps_venv"
                    )
                    python_path = _get_app_python(
                        app_name, wireless_version, desktop_app_daemon
                    )
                    
                    if use_uv:
                        install_cmd = [
                            "uv",
                            "pip",
                            "install",
                            "--python",
                            str(python_path),
                            "reachy-mini[gstreamer]",
                        ]
                    else:
                        install_cmd = [
                            str(python_path),
                            "-m",
                            "pip",
                            "install",
                            "reachy-mini[gstreamer]",
                        ]
                    
                    ret = await running_command(install_cmd, logger=logger)
                    if ret != 0:
                        logger.warning(
                            "Failed to pre-install reachy-mini, continuing anyway"
                        )
            else:
                logger.info(f"Using existing shared venv at {venv_path}")

            # Install package in the venv
            python_path = _get_app_python(
                app_name, wireless_version, desktop_app_daemon
            )
            
            if use_uv:
                install_cmd = ["uv", "pip", "install", "--python", str(python_path), target]
            else:
                install_cmd = [str(python_path), "-m", "pip", "install", target]
            
            ret = await running_command(install_cmd, logger=logger)

            if ret != 0:
                return ret

            logger.info(f"Successfully installed '{app_name}' in {venv_path}")
            success = True
            return 0
        finally:
            # Clean up broken venv on any failure (but not shared wireless venv)
            if (
                not success
                and venv_path.exists()
                and not (wireless_version and not desktop_app_daemon)
            ):
                logger.warning(f"Installation failed, cleaning up {venv_path}")
                shutil.rmtree(venv_path)
    else:
        # Original behavior: install into current environment
        if use_uv:
            install_cmd = ["uv", "pip", "install", "--python", sys.executable, target]
        else:
            install_cmd = [sys.executable, "-m", "pip", "install", target]
        
        return await running_command(install_cmd, logger=logger)


def get_app_module(
    app_name: str,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> str:
    """Get the module name for an app without loading it (for subprocess execution)."""
    if _should_use_separate_venvs(wireless_version, desktop_app_daemon):
        # Get module from separate venv's entry points
        site_packages = _get_app_site_packages(
            app_name, wireless_version, desktop_app_daemon
        )
        if not site_packages or not site_packages.exists():
            raise ValueError(f"App '{app_name}' venv not found or invalid")

        sys.path.insert(0, str(site_packages))
        try:
            eps = entry_points(group="reachy_mini_apps")
            ep = eps.select(name=app_name)
            if not ep:
                raise ValueError(f"No entry point found for app '{app_name}'")
            # Get module name without loading (e.g., "my_app.main" from "my_app.main:MyApp")
            return list(ep)[0].module
        finally:
            sys.path.pop(0)
    else:
        # Get module from current environment
        eps = entry_points(group="reachy_mini_apps", name=app_name)
        ep_list = list(eps)
        if not ep_list:
            raise ValueError(f"No entry point found for app '{app_name}'")
        return ep_list[0].module


async def uninstall_package(
    app_name: str,
    logger: logging.Logger,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> int:
    """Uninstall a package given an app name."""
    if _should_use_separate_venvs(wireless_version, desktop_app_daemon):
        venv_path = _get_app_venv_path(app_name, wireless_version, desktop_app_daemon)

        if not venv_path.exists():
            raise ValueError(f"Cannot uninstall app '{app_name}': it is not installed")

        if wireless_version and not desktop_app_daemon:
            # Wireless: shared venv, just uninstall the package
            logger.info(f"Uninstalling '{app_name}' from shared venv at {venv_path}")
            python_path = _get_app_python(
                app_name, wireless_version, desktop_app_daemon
            )
            
            # Check if uv is available
            use_uv = await _check_uv_available()
            
            if use_uv:
                uninstall_cmd = [
                    "uv",
                    "pip",
                    "uninstall",
                    "--python",
                    str(python_path),
                    "-y",
                    app_name,
                ]
            else:
                uninstall_cmd = [
                    str(python_path),
                    "-m",
                    "pip",
                    "uninstall",
                    "-y",
                    app_name,
                ]
            
            return await running_command(uninstall_cmd, logger=logger)
        else:
            # Desktop: remove the entire per-app venv directory
            logger.info(f"Removing venv for '{app_name}' at {venv_path}")
            shutil.rmtree(venv_path)
            logger.info(f"Successfully uninstalled '{app_name}'")
            return 0
    else:
        existing_apps = await list_available_apps()
        if app_name not in [app.name for app in existing_apps]:
            raise ValueError(f"Cannot uninstall app '{app_name}': it is not installed")

        # Check if uv is available
        use_uv = await _check_uv_available()
        
        logger.info(f"Removing package {app_name}")
        
        if use_uv:
            uninstall_cmd = ["uv", "pip", "uninstall", "--python", sys.executable, "-y", app_name]
        else:
            uninstall_cmd = [sys.executable, "-m", "pip", "uninstall", "-y", app_name]
        
        return await running_command(uninstall_cmd, logger=logger)