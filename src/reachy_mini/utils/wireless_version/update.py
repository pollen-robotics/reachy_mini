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

    Raises:
        RuntimeError: If the daemon venv install (or the final restart
            command) fails. The daemon is then NOT restarted, nothing has
            been applied, and the job is reported as failed - a retry stays
            possible. An apps_venv failure is deliberately non-fatal (see
            below).

    """
    # Update daemon venv. A failure here aborts the whole update BEFORE the
    # restart: the running daemon is untouched and the failed install left
    # the venv on the previous version, so a retry works.
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
        try:
            await call_logger_wrapper(cmd, logger, env=extra_env or None)
            logger.info("Apps venv SDK updated successfully")
        except RuntimeError as e:
            # Non-fatal by design. The daemon venv is already on the new
            # version at this point: aborting without restarting would leave
            # it installed-but-not-running, and a retry would be refused as
            # "no update available" (the availability check reads installed
            # metadata). Restart anyway; `check_and_sync_apps_venv_sdk`
            # re-syncs apps_venv to the daemon's version on every boot.
            logger.error(
                f"apps_venv SDK update failed (will re-sync on next boot): {e}"
            )
    else:
        logger.info("apps_venv not found, skipping")

    # Restart daemon to apply updates
    await call_logger_wrapper("sudo systemctl restart reachy-mini-daemon", logger)
