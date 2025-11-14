"""Hugging Face Spaces app source."""

import aiohttp
from pydantic import BaseModel

from .. import AppInfo, SourceKind


async def list_available_apps() -> list[AppInfo]:
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


async def app_info_from_space_url(space_url: str) -> AppInfo:
    """Get app info from a Hugging Face Space URL."""
    space_url = "/".join(space_url.rstrip("/").split("/")[-2:])
    url = f"https://huggingface.co/api/spaces/{space_url}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            item = await response.json()

    return AppInfo(
        name=item["id"].split("/")[-1],
        description=item["cardData"].get("short_description", ""),
        url=f"https://huggingface.co/spaces/{item['id']}",
        source_kind=SourceKind.HF_SPACE,
        extra=item,
    )


class DashboardAppList(BaseModel):
    """Model for dashboard app list."""

    dashboard_selected_apps: list[AppInfo]


async def get_dashboard_selection_apps() -> list[AppInfo]:
    """Get the list of apps selected for the dashboard."""
    dashboard_list_url = "https://huggingface.co/spaces/pollen-robotics/Reachy-Mini_Best_Spaces/raw/main/dashboard-app-list.json"

    async with aiohttp.ClientSession() as session:
        async with session.get(dashboard_list_url) as response:
            data = await response.text()

    apps = DashboardAppList.model_validate_json(data)
    return apps.dashboard_selected_apps
