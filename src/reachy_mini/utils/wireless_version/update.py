"""Module to handle software updates for the Reachy Mini wireless."""

import logging
import shutil
from pathlib import Path

from .utils import call_logger_wrapper

GITHUB_REPO = "pollen-robotics/reachy_mini"


async def update_reachy_mini(
    logger: logging.Logger,
    pre_release: bool = False,
    git_ref: str | None = None,
) -> None:
    """Update reachy_mini package and restart daemon.

    Args:
        logger: Logger for streaming output.
        pre_release: If True, install pre-release from PyPI (ignored if git_ref set).
        git_ref: If set, install from this GitHub tag/branch instead of PyPI.

    """
    # Build install command based on mode
    if git_ref:
        # Install from GitHub ref
        logger.info(f"Installing from GitHub ref: {git_ref}")
        git_url = f"git+https://github.com/{GITHUB_REPO}.git@{git_ref}"
        daemon_pkg = f"reachy_mini[wireless-version] @ {git_url}"
        apps_pkg = f"reachy-mini[gstreamer] @ {git_url}"
        extra_args = ["--force-reinstall"]
    else:
        # Install from PyPI
        logger.info("Installing from PyPI...")
        daemon_pkg = "reachy_mini[wireless-version]"
        apps_pkg = "reachy-mini[gstreamer]"
        extra_args = ["--pre"] if pre_release else []

    # Update daemon venv
    logger.info("Updating daemon venv...")
    await call_logger_wrapper(
        ["pip", "install", "--upgrade", daemon_pkg] + extra_args,
        logger,
    )

    # Update apps_venv if it exists
    apps_venv_python = Path("/venvs/apps_venv/bin/python")
    if apps_venv_python.exists():
        logger.info("Updating apps_venv SDK...")

        if shutil.which("uv"):
            install_cmd = [
                "uv", "pip", "install", "--python", str(apps_venv_python),
                "--upgrade", apps_pkg,
            ] + extra_args
        else:
            install_cmd = [
                str(Path("/venvs/apps_venv/bin/pip")),
                "install", "--upgrade", apps_pkg,
            ] + extra_args

        await call_logger_wrapper(install_cmd, logger)
        logger.info("Apps venv SDK updated successfully")
    else:
        logger.info("apps_venv not found, skipping")

    # Restart daemon to apply updates
    await call_logger_wrapper(
        ["sudo", "systemctl", "restart", "reachy-mini-daemon"], logger
    )
