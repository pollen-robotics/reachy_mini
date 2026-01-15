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


def _check_uv_available() -> bool:
    """Check if uv is available on the system."""
    return shutil.which("uv") is not None


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
                # Load saved metadata (e.g., private flag)
                metadata = _load_app_metadata(app_name)
                # Merge with current extra data
                extra_data = {
                    "custom_app_url": custom_app_url,
                    "venv_path": str(apps_venv),
                }
                extra_data.update(metadata)

                apps.append(
                    AppInfo(
                        name=app_name,
                        source_kind=SourceKind.INSTALLED,
                        extra=extra_data,
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

            # Load saved metadata (e.g., private flag)
            metadata = _load_app_metadata(app_name)
            # Merge with current extra data
            extra_data = {
                "custom_app_url": custom_app_url,
                "venv_path": str(venv_path),
            }
            extra_data.update(metadata)

            apps.append(
                AppInfo(
                    name=app_name,
                    source_kind=SourceKind.INSTALLED,
                    extra=extra_data,
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

        # Load saved metadata (searches by entry point name and HuggingFace space name)
        metadata = _find_metadata_for_entry_point(ep.name)
        # Merge with current extra data
        extra_data = {"custom_app_url": custom_app_url}
        extra_data.update(metadata)

        apps.append(
            AppInfo(
                name=ep.name,
                source_kind=SourceKind.INSTALLED,
                extra=extra_data,
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


def _get_app_metadata_path(app_name: str) -> Path:
    """Get the path to the metadata file for an app."""
    parent_dir = _get_venv_parent_dir()
    metadata_dir = parent_dir / ".app_metadata"
    metadata_dir.mkdir(exist_ok=True)
    return metadata_dir / f"{app_name}.json"


def _save_app_metadata(app_name: str, metadata: dict) -> None:  # type: ignore
    """Save metadata for an app."""
    import json

    metadata_path = _get_app_metadata_path(app_name)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)


def _load_app_metadata(app_name: str) -> dict:  # type: ignore
    """Load metadata for an app."""
    import json

    metadata_path = _get_app_metadata_path(app_name)
    if not metadata_path.exists():
        return {}
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)  # type: ignore
    except Exception:
        return {}


def _find_metadata_for_entry_point(ep_name: str) -> dict:  # type: ignore
    """Find metadata for an entry point, even if saved with a different name.

    When apps are installed from HuggingFace, metadata is saved with the space name,
    but entry points may use a different Python package name (different separators
    or additional suffixes). This function searches for matching metadata.

    Returns:
        dict: The loaded metadata (may be empty if not found)

    """
    import json

    # First, try direct match by entry point name
    metadata = _load_app_metadata(ep_name)
    if metadata:
        return metadata

    # If not found, scan all metadata files to find a match
    parent_dir = _get_venv_parent_dir()
    metadata_dir = parent_dir / ".app_metadata"

    if not metadata_dir.exists():
        return {}

    # Normalize name for comparison (remove underscores/dashes, lowercase)
    def normalize(name: str) -> str:
        return name.lower().replace("_", "").replace("-", "")

    ep_normalized = normalize(ep_name)

    for metadata_file in metadata_dir.glob("*.json"):
        try:
            with open(metadata_file, "r") as f:
                file_metadata = json.load(f)

            # Get the space name from the file (filename without .json)
            space_name = metadata_file.stem

            # Check 1: Normalized name match
            if normalize(space_name) == ep_normalized:
                return file_metadata  # type: ignore

            # Check 2: Entry point name appears in siblings (package structure)
            siblings = file_metadata.get("siblings", [])
            for sibling in siblings:
                rfilename = sibling.get("rfilename", "")
                # Check if entry point package folder exists in siblings
                if rfilename.startswith(f"{ep_name}/"):
                    return file_metadata  # type: ignore

            # Check 3: extra.id contains normalized match
            extra_id = file_metadata.get("id", "")
            if extra_id:
                # Extract app name from full ID (remove author prefix)
                id_name = extra_id.split("/")[-1] if "/" in extra_id else extra_id
                if normalize(id_name) == ep_normalized:
                    return file_metadata  # type: ignore

        except Exception:
            continue

    # No match found
    return {}


def _delete_app_metadata(app_name: str) -> None:
    """Delete metadata for an app."""
    metadata_path = _get_app_metadata_path(app_name)
    if metadata_path.exists():
        metadata_path.unlink()


async def install_package(
    app: AppInfo,
    logger: logging.Logger,
    wireless_version: bool = False,
    desktop_app_daemon: bool = False,
) -> int:
    """Install a package given an AppInfo object, streaming logs."""
    # Check if uv is available
    use_uv = _check_uv_available()
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

        # Check if this is a private space installation
        is_private = app.extra.get("private", False)
        token = None

        if is_private:
            # Get token for private spaces
            from reachy_mini.apps.sources import hf_auth

            token = hf_auth.get_hf_token()
            if not token:
                logger.error("Private space requires authentication but no token found")
                return 1
            logger.info("Using stored HF token for private space access")

        try:
            # First, verify the space exists and we have access
            from huggingface_hub import HfApi

            try:
                api = HfApi(token=token)
                space_info = api.space_info(repo_id=repo_id)
                logger.info(
                    f"Space found: {space_info.id} (private={space_info.private})"
                )

                # List all files in the space to see what's available
                try:
                    files_in_repo = api.list_repo_files(
                        repo_id=repo_id, repo_type="space", token=token
                    )
                    logger.info(f"Files available in space: {', '.join(files_in_repo)}")
                except Exception as list_error:
                    logger.warning(f"Could not list files in space: {list_error}")
            except Exception as verify_error:
                logger.error(f"Cannot access space {repo_id}: {verify_error}")
                if "404" in str(verify_error):
                    logger.error(
                        f"Space '{repo_id}' not found. Please check the space ID and your permissions."
                    )
                elif "401" in str(verify_error) or "403" in str(verify_error):
                    logger.error(
                        f"Access denied to space '{repo_id}'. Please check your HuggingFace token permissions."
                    )
                return 1

            # Download the space
            logger.info("Attempting to download all files from space...")
            # For private spaces, we need to be careful about missing files like .gitattributes
            # snapshot_download can fail on 404 for optional git metadata files
            target = await asyncio.to_thread(
                snapshot_download,
                repo_id=repo_id,
                repo_type="space",
                token=token,
                ignore_patterns=[
                    ".gitattributes",
                    ".gitignore",
                ],  # Skip git metadata that may 404
                allow_patterns=None,  # Download all other files
            )
            logger.info(f"Downloaded to: {target}")

            # Check what files were downloaded to help with debugging
            import os

            downloaded_files = []
            for root, dirs, files in os.walk(target):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), target)
                    downloaded_files.append(rel_path)
            logger.info(f"Downloaded files: {', '.join(downloaded_files)}")

            # Check if this looks like a Python package
            has_pyproject = os.path.exists(os.path.join(target, "pyproject.toml"))
            has_setup = os.path.exists(os.path.join(target, "setup.py"))

            if not has_pyproject and not has_setup:
                logger.warning(
                    f"Space does not appear to have pyproject.toml or setup.py in the root directory. "
                    f"For a Reachy Mini app, you need a proper Python package structure. "
                    f"Downloaded files: {', '.join(downloaded_files)}"
                )
                logger.info(
                    "If your package files are in a subdirectory, make sure they're in the root of the space. "
                    "Check that pyproject.toml or setup.py is committed to your HuggingFace Space."
                )
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                logger.error(
                    f"Authentication failed: {e}\n"
                    "Please check that your HuggingFace token has access to this space."
                )
            elif "404" in error_msg:
                logger.error(
                    f"Space not found: {e}\n"
                    f"Please verify that '{repo_id}' exists and you have access."
                )
            else:
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
                install_cmd = [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python_path),
                    target,
                ]
            else:
                install_cmd = [str(python_path), "-m", "pip", "install", target]

            ret = await running_command(install_cmd, logger=logger)

            if ret != 0:
                return ret

            logger.info(f"Successfully installed '{app_name}' in {venv_path}")
            success = True

            # Save app metadata (e.g., private flag)
            if app.extra:
                _save_app_metadata(app_name, app.extra)
                logger.info(f"Saved metadata for '{app_name}': {app.extra}")

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

        ret = await running_command(install_cmd, logger=logger)

        if ret == 0 and app.extra:
            # Save app metadata so we can match by extra.id later
            # Use the space name (app.name) as the key
            _save_app_metadata(app.name, app.extra)
            logger.info(f"Saved metadata for '{app.name}': {app.extra}")

        return ret


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
            use_uv = _check_uv_available()

            if use_uv:
                uninstall_cmd = [
                    "uv",
                    "pip",
                    "uninstall",
                    "--python",
                    str(python_path),
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

            ret = await running_command(uninstall_cmd, logger=logger)
            if ret == 0:
                logger.info(f"Successfully uninstalled '{app_name}'")
                # Delete app metadata
                _delete_app_metadata(app_name)
            return ret
        else:
            # Desktop: remove the entire per-app venv directory
            logger.info(f"Removing venv for '{app_name}' at {venv_path}")
            shutil.rmtree(venv_path)
            logger.info(f"Successfully uninstalled '{app_name}'")
            # Delete app metadata
            _delete_app_metadata(app_name)
            return 0
    else:
        existing_apps = await list_available_apps()
        if app_name not in [app.name for app in existing_apps]:
            raise ValueError(f"Cannot uninstall app '{app_name}': it is not installed")

        # Check if uv is available
        use_uv = _check_uv_available()

        logger.info(f"Removing package {app_name}")

        if use_uv:
            uninstall_cmd = [
                "uv",
                "pip",
                "uninstall",
                "--python",
                sys.executable,
                app_name,
            ]
        else:
            uninstall_cmd = [sys.executable, "-m", "pip", "uninstall", "-y", app_name]

        ret = await running_command(uninstall_cmd, logger=logger)
        if ret == 0:
            logger.info(f"Successfully uninstalled '{app_name}'")
            # Delete app metadata
            _delete_app_metadata(app_name)
        return ret
