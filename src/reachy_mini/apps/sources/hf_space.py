"""Hugging Face Spaces app source."""

import asyncio
import json
import logging

import aiohttp
from huggingface_hub import HfApi

from .. import AppInfo, SourceKind
from . import hf_auth

# Constants
AUTHORIZED_APP_LIST_URL = "https://huggingface.co/datasets/pollen-robotics/reachy-mini-official-app-store/raw/main/app-list.json"
HF_SPACES_API_URL = "https://huggingface.co/api/spaces"
# TODO look for js apps too (reachy_mini_js_app)
HF_SPACES_FILTER = "reachy_mini_python_app"
HF_SPACES_LIMIT = 500
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)
logger = logging.getLogger("reachy_mini.apps.sources.hf_space")
SpaceData = dict[str, object]


def _coerce_space_data(value: object) -> SpaceData | None:
    """Return a string-keyed dict for space payloads."""
    if not isinstance(value, dict):
        return None
    return {str(key): item for key, item in value.items()}


def _coerce_space_list(value: object) -> list[SpaceData]:
    """Return only dict-shaped items from a raw HF payload."""
    if not isinstance(value, list):
        return []

    spaces: list[SpaceData] = []
    for item in value:
        space_data = _coerce_space_data(item)
        if space_data is not None:
            spaces.append(space_data)
    return spaces


def _get_string(item: SpaceData, key: str) -> str | None:
    """Read a string field from a space payload."""
    value = item.get(key)
    return value if isinstance(value, str) else None


def _get_card_data(item: SpaceData) -> SpaceData:
    """Read card data from a space payload."""
    card_data = item.get("cardData")
    return card_data if isinstance(card_data, dict) else {}


def _normalize_space_data(space_data: SpaceData) -> SpaceData:
    """Normalize HF API responses to the shape used by the app store."""
    normalized = dict(space_data)

    created_at = normalized.pop("created_at", None)
    if not normalized.get("createdAt") and created_at is not None:
        normalized["createdAt"] = created_at

    last_modified = normalized.pop("last_modified", None)
    if not normalized.get("lastModified") and last_modified is not None:
        normalized["lastModified"] = last_modified

    card_data = normalized.pop("card_data", None)
    if not normalized.get("cardData") and card_data is not None:
        normalized["cardData"] = card_data

    return normalized


def _build_app_info(item: SpaceData | None) -> AppInfo | None:
    """Build AppInfo from a normalized Hugging Face Space payload."""
    if item is None:
        return None

    item = _normalize_space_data(item)
    space_id = _get_string(item, "id")
    if space_id is None:
        return None
    card_data = _get_card_data(item)
    short_description = _get_string(card_data, "short_description") or ""

    return AppInfo(
        name=space_id.split("/")[-1],
        description=short_description,
        url=f"https://huggingface.co/spaces/{space_id}",
        source_kind=SourceKind.HF_SPACE,
        extra=item,
    )


def _list_all_spaces_with_hf_api(token: str | None) -> list[SpaceData]:
    """List spaces with Hugging Face Hub API using an optional token."""
    api = HfApi()
    spaces = api.list_spaces(
        filter=HF_SPACES_FILTER,
        sort="likes",
        limit=HF_SPACES_LIMIT,
        full=True,
        token=token,
    )
    payloads: list[SpaceData] = []
    for space in spaces:
        space_data = _coerce_space_data(space.__dict__)
        if space_data is None or _get_string(space_data, "id") is None:
            continue
        payloads.append(_normalize_space_data(space_data))
    return payloads


async def _fetch_space_data(
    session: aiohttp.ClientSession, space_id: str
) -> SpaceData | None:
    """Fetch data for a single space from Hugging Face API."""
    url = f"{HF_SPACES_API_URL}/{space_id}"
    try:
        async with session.get(url, timeout=REQUEST_TIMEOUT) as response:
            if response.status == 200:
                return _coerce_space_data(await response.json())
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
    try:
        data = await asyncio.to_thread(_list_all_spaces_with_hf_api, token)
    except Exception as exc:
        logger.warning("Could not list HF Spaces: %s", exc)
        return []

    apps = []
    for item in data:
        app_info = _build_app_info(item)
        if app_info is not None:
            apps.append(app_info)

    return apps
