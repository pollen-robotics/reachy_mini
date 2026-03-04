"""Module to handle software updates for the Reachy Mini wireless."""

import logging
from pathlib import Path

from .utils import build_install_command, call_logger_wrapper


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
    # Update daemon venv
    logger.info("Updating daemon venv...")
    cmd, extra_env = build_install_command(
        extras="wireless-version",
        git_ref=git_ref, 
        pre_release=pre_release, 
        upgrade=True,
    )
    await call_logger_wrapper(cmd, logger, env=extra_env or None)

    # Update apps_venv if it exists
    apps_venv_python = Path("/venvs/apps_venv/bin/python")
    if apps_venv_python.exists():
        logger.info("Updating apps_venv SDK...")
        cmd, extra_env = build_install_command( 
            extras="",
            git_ref=git_ref, 
            pre_release=pre_release,
            python=apps_venv_python, 
            upgrade=True,
        )
        await call_logger_wrapper(cmd, logger, env=extra_env or None)
        logger.info("Apps venv SDK updated successfully")
    else:
        logger.info("apps_venv not found, skipping")

    # Restart daemon to apply updates
    await call_logger_wrapper("sudo systemctl restart reachy-mini-daemon", logger)
