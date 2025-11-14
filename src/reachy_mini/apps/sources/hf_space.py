"""Hugging Face Spaces app source."""

import asyncio
import json
from typing import Any, Dict

import aiohttp

from .. import AppInfo, SourceKind


async def _fetch_space_data(
    session: aiohttp.ClientSession, space_id: str
) -> Dict[str, Any] | None:
    """Fetch data for a single space from Hugging Face API."""
    url = f"https://huggingface.co/api/spaces/{space_id}"
    try:
        async with session.get(url) as response:
            if response.status == 200:
                data: Dict[str, Any] = await response.json()
                return data
            else:
                return None
    except Exception:
        return None


async def list_available_apps() -> list[AppInfo]:
    """List apps available on Hugging Face Spaces."""
    authorized_list_url = "https://huggingface.co/datasets/pollen-robotics/reachy-mini-official-app-store/raw/main/app-list.json"

    async with aiohttp.ClientSession() as session:
        # Fetch the list of authorized app IDs
        try:
            async with session.get(authorized_list_url) as response:
                response.raise_for_status()
                text = await response.text()
                authorized_ids = json.loads(text)
        except (aiohttp.ClientError, json.JSONDecodeError):
            return []

        if not isinstance(authorized_ids, list):
            return []

        # Fetch data for each space in parallel
        tasks = [_fetch_space_data(session, space_id) for space_id in authorized_ids]
        spaces_data = await asyncio.gather(*tasks)

        # Build AppInfo list from fetched data
        apps = []
        for item in spaces_data:
            if item is None or "id" not in item:
                continue

            apps.append(
                AppInfo(
                    name=item["id"].split("/")[-1],
                    description=item.get("cardData", {}).get("short_description", ""),
                    url=f"https://huggingface.co/spaces/{item['id']}",
                    source_kind=SourceKind.HF_SPACE,
                    extra=item,
                )
            )

        return apps
