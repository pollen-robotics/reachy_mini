"""Module to handle software updates for the Reachy Mini wireless."""

import logging

from .utils import call_logger_wrapper


async def update_reachy_mini(pre_release: bool, logger: logging.Logger) -> None:
    """Perform a software update by upgrading the reachy_mini package and restarting the daemon."""
    extra_args = []
    if pre_release:
        extra_args.append("--pre")

    await call_logger_wrapper(
        ["pip", "install", "--upgrade", "reachy_mini[wireless-version]"] + extra_args,
        logger,
    )
    await call_logger_wrapper(
        ["sudo", "systemctl", "restart", "reachy-mini-daemon"], logger
    )
