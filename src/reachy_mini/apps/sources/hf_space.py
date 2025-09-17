"""Hugging Face Spaces app source."""

import aiohttp

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
