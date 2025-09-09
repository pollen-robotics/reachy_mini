"""App management for Reachy Mini."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum

import aiohttp

from . import common_venv


class SourceKind(str, Enum):
    """Kinds of app source."""

    HF_SPACE = "hf_space"
    INSTALLED = "installed"


@dataclass
class AppInfo:
    """Metadata about an app."""

    name: str
    source_kind: SourceKind
    description: str = ""
    url: str | None = None
    extra: dict = field(default_factory=dict)


# TODO: Use pipx for a cleaner app installation and management?


class AppManager:
    """Manager for Reachy Mini apps."""

    # Apps management interface
    async def list_available_apps(self) -> list[AppInfo]:
        """List available apps (parallel async)."""
        coros = [getattr(self, f"list_{kind.value}_apps")() for kind in SourceKind]
        results = await asyncio.gather(*coros)
        return sum(results, [])

    async def list_hf_space_apps(self) -> list[AppInfo]:
        """List apps available on Hugging Face Spaces."""
        url = "https://huggingface.co/api/spaces?filter=reachy_mini&sort=likes&direction=-1&limit=50&full=true"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()
        apps = []
        for item in data:
            apps.append(
                AppInfo(
                    name=item["id"].split("/")[-1],
                    description=item["cardData"].get("short_description", ""),
                    url=f"https://huggingface.co/spaces/{item['id']}",
                    source_kind=SourceKind.HF_SPACE,
                    extra=item,
                )
            )
        return apps

    async def list_installed_apps(self) -> list[AppInfo]:
        """List installed apps."""
        return common_venv.list_apps_from_entry_points("reachy_mini_apps")

    async def install_new_app(self, app: AppInfo) -> None:
        """Install a new app by name."""
        await common_venv.install_package(app)

    async def remove_app(self, app: AppInfo) -> None:
        """Remove an installed app by name."""
        await common_venv.uninstall_package(app)

    async def update_app(self, app: AppInfo) -> None:
        """Update an installed app by name."""
        await common_venv.update_package(app)

    # App lifecycle management
    async def start_app(self, app: AppInfo):
        """Start the app."""

    async def stop_app(self, app: AppInfo):
        """Stop the app."""

    async def restart_app(self, app: AppInfo):
        """Restart the app."""

    async def app_status(self, app: AppInfo) -> dict:
        """Get the current status of the app."""
        return {"status": "running"}
