"""Module to handle software updates for the Reachy Mini wireless."""

import logging
from pathlib import Path

from . import startup_check
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
        RuntimeError: If the daemon venv install or the restart command
            fails. An apps_venv failure is deliberately non-fatal.

    """
    # Invalidate the startup-check stamp before touching any venv, so a crash
    # mid-update always triggers the full startup checks on the next boot.
    # (The stamp's dist-info signature would catch this anyway; this makes it
    # double-covered.)
    try:
        startup_check.clear_startup_stamp()
    except OSError:
        logger.error(
            "Could not clear the startup-check stamp at %s. Continuing: the "
            "install signature will invalidate it on the next boot anyway. "
            "A stamp that cannot be deleted points at a filesystem or "
            "ownership problem worth investigating.",
            startup_check.STAMP_PATH,
            exc_info=True,
        )

    # Update daemon venv. Fatal: abort before the restart. On the PyPI path
    # the venv is left on the previous version, so a retry works.
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
        except Exception as e:
            # Non-fatal: the daemon venv is already on the new version, and
            # check_and_sync_apps_venv_sdk re-syncs apps_venv on the next boot.
            logger.error(
                f"apps_venv SDK update failed (will re-sync on next boot): {e}"
            )
    else:
        logger.info("apps_venv not found, skipping")

    # Restart daemon to apply updates. Two constraints, both load-bearing:
    # - the argv must match /etc/sudoers.d/010-pollen-reachy exactly (any extra
    #   flag, --no-block included, makes sudo demand a password and fail);
    # - systemd SIGTERMs our own cgroup, killing this systemctl child, so
    #   143/-15 means "the restart we asked for is happening", not a failure.
    # Everything else (unknown unit, sudo refusal) still raises.
    await call_logger_wrapper(
        "sudo systemctl restart reachy-mini-daemon",
        logger,
        ok_returncodes=(0, -15, 143),
    )
