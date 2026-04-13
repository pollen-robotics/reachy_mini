"""Hugging Face Spaces app source."""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import aiohttp

from .. import AppInfo, SourceKind

logger = logging.getLogger(__name__)

# Constants
AUTHORIZED_APP_LIST_URL = "https://huggingface.co/datasets/pollen-robotics/reachy-mini-official-app-store/raw/main/app-list.json"
HF_SPACES_API_URL = "https://huggingface.co/api/spaces"
# TODO look for js apps too (reachy_mini_js_app)
HF_SPACES_FILTER_URL = "https://huggingface.co/api/spaces?filter=reachy_mini_python_app&sort=likes&direction=-1&limit=500&full=true"
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)


def _get_auth_headers() -> dict[str, str]:
    """Return Authorization header if an HF token is available."""
    from . import hf_auth

    token = hf_auth.get_hf_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


async def _fetch_space_data(
    session: aiohttp.ClientSession,
    space_id: str,
    headers: Optional[dict[str, str]] = None,
) -> Dict[str, Any] | None:
    """Fetch data for a single space from Hugging Face API."""
    url = f"{HF_SPACES_API_URL}/{space_id}"
    try:
        async with session.get(
            url, timeout=REQUEST_TIMEOUT, headers=headers
        ) as response:
            if response.status == 200:
                data: Dict[str, Any] = await response.json()
                return data
            else:
                return None
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return None


async def list_available_apps() -> list[AppInfo]:
    """List apps available on Hugging Face Spaces."""
    auth_headers = _get_auth_headers()

    async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
        # Fetch the list of authorized app IDs
        try:
            async with session.get(AUTHORIZED_APP_LIST_URL) as response:
                response.raise_for_status()
                text = await response.text()
                authorized_ids = json.loads(text)
        except (aiohttp.ClientError, json.JSONDecodeError):
            return []

        if not isinstance(authorized_ids, list):
            return []

        # Filter to only string elements
        authorized_ids = [
            space_id for space_id in authorized_ids if isinstance(space_id, str)
        ]

        # Fetch data for each space in parallel (authenticated to include private spaces)
        tasks = [
            _fetch_space_data(session, space_id, auth_headers)
            for space_id in authorized_ids
        ]
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


async def list_all_apps() -> list[AppInfo]:
    """List all apps available on Hugging Face Spaces (including unofficial ones).

    When an HF token is available, the request is authenticated so the
    API also returns private spaces the user has access to.
    """
    auth_headers = _get_auth_headers()

    async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
        try:
            async with session.get(
                HF_SPACES_FILTER_URL, headers=auth_headers
            ) as response:
                response.raise_for_status()
                data: list[Dict[str, Any]] = await response.json()
        except (aiohttp.ClientError, json.JSONDecodeError, asyncio.TimeoutError):
            return []

        if not isinstance(data, list):
            return []

        apps = []
        for item in data:
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
