"""Hugging Face Spaces app source."""

import asyncio
import json
import logging
from typing import Any, Dict

import aiohttp
from huggingface_hub import HfApi

from .. import AppInfo, SourceKind
from . import hf_auth

# Constants
AUTHORIZED_APP_LIST_URL = "https://huggingface.co/datasets/pollen-robotics/reachy-mini-official-app-store/raw/main/app-list.json"
HF_SPACES_API_URL = "https://huggingface.co/api/spaces"
# TODO look for js apps too (reachy_mini_js_app)
HF_SPACES_FILTER = "reachy_mini_python_app"
HF_SPACES_FILTER_URL = f"https://huggingface.co/api/spaces?filter={HF_SPACES_FILTER}&sort=likes&direction=-1&limit=500&full=true"
HF_SPACES_LIMIT = 500
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)
logger = logging.getLogger("reachy_mini.apps.sources.hf_space")


def _normalize_space_data(space_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize HF API responses to the shape used by the app store."""
    normalized = dict(space_data)

    created_at = normalized.get("createdAt") or normalized.get("created_at")
    if created_at is not None:
        normalized["createdAt"] = created_at

    last_modified = normalized.get("lastModified") or normalized.get("last_modified")
    if last_modified is not None:
        normalized["lastModified"] = last_modified

    card_data = normalized.get("cardData") or normalized.get("card_data")
    if card_data is not None:
        normalized["cardData"] = card_data

    return normalized


def _build_app_info(item: Dict[str, Any]) -> AppInfo | None:
    """Build AppInfo from a normalized Hugging Face Space payload."""
    if item is None or "id" not in item:
        return None

    item = _normalize_space_data(item)

    return AppInfo(
        name=item["id"].split("/")[-1],
        description=item.get("cardData", {}).get("short_description", ""),
        url=f"https://huggingface.co/spaces/{item['id']}",
        source_kind=SourceKind.HF_SPACE,
        extra=item,
    )


def _list_all_spaces_with_hf_api(token: str) -> list[Dict[str, Any]]:
    """List spaces with Hugging Face Hub API using the stored token."""
    api = HfApi(token=token)
    spaces = api.list_spaces(
        filter=HF_SPACES_FILTER,
        sort="likes",
        limit=HF_SPACES_LIMIT,
        full=True,
        token=token,
    )
    return [
        _normalize_space_data(space.__dict__)
        for space in spaces
        if getattr(space, "id", None)
    ]


async def _fetch_space_data(
    session: aiohttp.ClientSession, space_id: str
) -> Dict[str, Any] | None:
    """Fetch data for a single space from Hugging Face API."""
    url = f"{HF_SPACES_API_URL}/{space_id}"
    try:
        async with session.get(url, timeout=REQUEST_TIMEOUT) as response:
            if response.status == 200:
                data: Dict[str, Any] = await response.json()
                return data
            else:
                return None
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return None


async def list_available_apps() -> list[AppInfo]:
    """List apps available on Hugging Face Spaces."""
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

        # Fetch data for each space in parallel
        tasks = [_fetch_space_data(session, space_id) for space_id in authorized_ids]
        spaces_data = await asyncio.gather(*tasks)

        # Build AppInfo list from fetched data
        apps = []
        for item in spaces_data:
            app_info = _build_app_info(item)
            if app_info is not None:
                apps.append(app_info)

        return apps


async def list_all_apps() -> list[AppInfo]:
    """List all apps available on Hugging Face Spaces (including private ones when authenticated)."""
    token = hf_auth.get_hf_token()
    if token:
        try:
            data = await asyncio.to_thread(_list_all_spaces_with_hf_api, token)
            apps = []
            for item in data:
                app_info = _build_app_info(item)
                if app_info is not None:
                    apps.append(app_info)
            return apps
        except Exception as exc:
            logger.warning("Falling back to the public HF catalog: %s", exc)

    async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
        try:
            async with session.get(HF_SPACES_FILTER_URL) as response:
                response.raise_for_status()
                data: list[Dict[str, Any]] = await response.json()
        except (aiohttp.ClientError, json.JSONDecodeError, asyncio.TimeoutError):
            return []

        if not isinstance(data, list):
            return []

        apps = []
        for item in data:
            app_info = _build_app_info(item)
            if app_info is not None:
                apps.append(app_info)

        return apps
