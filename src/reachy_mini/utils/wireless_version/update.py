"""Module to handle software updates for the Reachy Mini wireless."""

import logging

from .utils import call


async def update_reachy_mini(logger: logging.Logger) -> None:
    """Perform a software update by upgrading the reachy_mini package and restarting the daemon."""
    await call(["pip", "install", "--upgrade", "reachy_mini[wireless-version]"], logger)
    await call(["sudo", "systemctl", "restart", "reachy-mini-daemon"], logger)
