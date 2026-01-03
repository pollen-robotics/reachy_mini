"""Module to handle software updates for the Reachy Mini wireless."""

import logging
import shutil
from pathlib import Path

from .utils import call_logger_wrapper


async def update_reachy_mini(pre_release: bool, logger: logging.Logger) -> None:
    """Perform a software update by upgrading the reachy_mini package and restarting the daemon.

    This updates both:
    - The daemon venv (where the daemon runs)
    - The apps_venv (where apps run) - to keep SDK in sync
    """
    extra_args = []
    if pre_release:
        extra_args.append("--pre")

    # Update daemon venv
    logger.info("Updating daemon venv...")
    await call_logger_wrapper(
        ["pip", "install", "--upgrade", "reachy_mini[wireless-version]"] + extra_args,
        logger,
    )

    # Update apps_venv if it exists
    apps_venv_python = Path("/venvs/apps_venv/bin/python")
    if apps_venv_python.exists():
        logger.info("Updating apps_venv SDK...")

        # Use uv if available for faster installs
        use_uv = shutil.which("uv") is not None

        if use_uv:
            install_cmd = [
                "uv",
                "pip",
                "install",
                "--python",
                str(apps_venv_python),
                "--upgrade",
                "reachy-mini[gstreamer]",
            ] + extra_args
        else:
            apps_venv_pip = Path("/venvs/apps_venv/bin/pip")
            install_cmd = [
                str(apps_venv_pip),
                "install",
                "--upgrade",
                "reachy-mini[gstreamer]",
            ] + extra_args

        await call_logger_wrapper(install_cmd, logger)
        logger.info("Apps venv SDK updated successfully")
    else:
        logger.info("apps_venv not found, skipping apps venv update")

    # Restart daemon to apply updates
    await call_logger_wrapper(
        ["sudo", "systemctl", "restart", "reachy-mini-daemon"], logger
    )
