"""Reflash the Reachy Mini wireless OS (CM4 eMMC) with a ReachyMiniOS release, over WiFi.

The work is done by the scripts in ``os_reflash/`` (see its README): this
module only resolves a release of pollen-robotics/reachy-mini-os to its image
URL and checksum, and runs the upgrade script as root. Only official releases
can be flashed: the daemon API never takes an arbitrary image URL.
"""

import logging
import shlex
from pathlib import Path

import requests

from .utils import call_logger_wrapper

OS_REPO = "pollen-robotics/reachy-mini-os"
UPGRADE_SCRIPT = Path(__file__).parent / "os_reflash" / "reachy-os-upgrade.sh"


class OSReleaseError(RuntimeError):
    """A ReachyMiniOS release cannot be found or has no image."""


def get_os_release(tag: str | None = None) -> dict[str, str | None]:
    """Find a ReachyMiniOS release on GitHub (the latest one when *tag* is None).

    Returns:
        ``tag``, ``zip_url`` and ``sha256`` (``None`` when the release does not
        publish a ``<image>.zip.sha256`` asset).

    Raises:
        OSReleaseError: If the release does not exist or has no ``.zip`` image.

    """
    path = f"tags/{tag}" if tag else "latest"
    resp = requests.get(
        f"https://api.github.com/repos/{OS_REPO}/releases/{path}", timeout=10
    )
    if resp.status_code == 404:
        raise OSReleaseError(f"No ReachyMiniOS release {tag or 'published'}")
    resp.raise_for_status()
    release = resp.json()
    assets = {a["name"]: a["browser_download_url"] for a in release["assets"]}
    zips = sorted(name for name in assets if name.endswith(".zip"))
    if not zips:
        raise OSReleaseError(f"Release {release['tag_name']} has no .zip image")

    sha256 = None
    sha_url = assets.get(zips[0] + ".sha256")
    if sha_url is not None:
        sha_resp = requests.get(sha_url, timeout=10)
        sha_resp.raise_for_status()
        sha256 = sha_resp.text.split()[0]

    return {"tag": release["tag_name"], "zip_url": assets[zips[0]], "sha256": sha256}


async def reflash_os(
    logger: logging.Logger, tag: str | None = None, dry_run: bool = False
) -> None:
    """Reflash the robot with a ReachyMiniOS release, then reboot into it.

    The upgrade script prepares a one-shot flasher and reboots into it, so on
    success this never returns: the daemon is stopped by the reboot. With
    *dry_run* the flasher does everything but the write (WiFi, download,
    checks) and reboots into the current OS.

    Raises:
        OSReleaseError: If the release cannot be found.
        RuntimeError: If the script fails, e.g. when sudo needs a password:
            on ReachyMiniOS v1.0.x pollen may not run it as root.

    """
    release = get_os_release(tag)
    logger.info(f"Reflashing with ReachyMiniOS {release['tag']} (dry run: {dry_run})")
    if release["sha256"] is None:
        logger.warning("Release publishes no sha256, only the zip CRC is checked")

    args = ["sudo", "-n", "/usr/bin/bash", str(UPGRADE_SCRIPT), "--yes"]
    if dry_run:
        args.append("--dry-run")
    if release["sha256"]:
        args += ["--sha256", release["sha256"]]
    args.append(str(release["zip_url"]))

    # The script ends with a reboot, which SIGTERMs the daemon's cgroup.
    await call_logger_wrapper(shlex.join(args), logger, ok_returncodes=(0, -15, 143))
