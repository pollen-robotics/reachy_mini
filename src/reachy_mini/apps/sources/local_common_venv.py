"""Utilities for local common venv apps source."""

import asyncio
import logging
import sys
from importlib.metadata import entry_points

from huggingface_hub import snapshot_download

from .. import AppInfo, SourceKind
from ..utils import running_command


async def list_available_apps() -> list[AppInfo]:
    """List apps available from entry points."""
    entry_point_apps = list(entry_points(group="reachy_mini_apps"))

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


async def install_package(app: AppInfo, logger: logging.Logger) -> int:
    """Install a package given an AppInfo object, streaming logs."""
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

    return await running_command(
        [sys.executable, "-m", "pip", "install", target],
        logger=logger,
    )


async def uninstall_package(app_name: str, logger: logging.Logger) -> int:
    """Uninstall a package given an app name."""
    existing_apps = await list_available_apps()
    if app_name not in [app.name for app in existing_apps]:
        raise ValueError(f"Cannot uninstall app '{app_name}': it is not installed")

    return await running_command(
        [sys.executable, "-m", "pip", "uninstall", "-y", app_name],
        logger=logger,
    )
