"""Hugging Face Spaces app source."""

import asyncio
import json
import logging

import aiohttp

from reachy_mini.utils.proxy import proxy_for

from .. import AppInfo, SourceKind
from . import hf_auth

# Constants
AUTHORIZED_APP_LIST_URL = "https://huggingface.co/datasets/pollen-robotics/reachy-mini-official-app-store/raw/main/app-list.json"
HF_SPACES_API_URL = "https://huggingface.co/api/spaces"
# TODO look for js apps too (reachy_mini_js_app)
HF_SPACES_FILTER = "reachy_mini_python_app"
HF_SPACES_LIMIT = 500
# Only the fields the app store clients read from AppInfo.extra. Requesting
# them explicitly (instead of full=True) leaves out each space's file list
# ("siblings"), which is ~90% of the payload and is never read from the catalog.
HF_SPACE_EXPAND_FIELDS = [
    "author",
    "cardData",
    "createdAt",
    "lastModified",
    "likes",
    "private",
    "runtime",
    "sdk",
    "tags",
]
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

    # The file list is never read from the catalog; drop it so neither path
    # (HfApi listing or per-space HTTP fetch) ships it to clients.
    normalized.pop("siblings", None)

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


async def _fetch_all_spaces(
    session: aiohttp.ClientSession, token: str | None
) -> list[SpaceData]:
    """List spaces over plain HTTP, following pagination links.

    The HTTP API already returns the camelCase JSON the app store expects, so
    this avoids materializing HfApi objects and converting them back (which
    costs seconds on a CM4 for a few hundred spaces).
    """
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    params: list[tuple[str, str]] = [
        ("filter", HF_SPACES_FILTER),
        ("sort", "likes"),
        ("limit", str(HF_SPACES_LIMIT)),
        *(("expand", field) for field in HF_SPACE_EXPAND_FIELDS),
    ]

    spaces: list[SpaceData] = []
    url: str | None = HF_SPACES_API_URL
    while url:
        async with session.get(
            url,
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            proxy=proxy_for(url),
        ) as response:
            response.raise_for_status()
            spaces.extend(_coerce_space_list(await response.json()))
            next_link = response.links.get("next")
            url = str(next_link["url"]) if next_link else None
            params = []  # the next link already carries the query
    return [_normalize_space_data(space) for space in spaces]


async def _fetch_space_data(
    session: aiohttp.ClientSession, space_id: str
) -> SpaceData | None:
    """Fetch data for a single space from Hugging Face API."""
    url = f"{HF_SPACES_API_URL}/{space_id}"
    try:
        async with session.get(
            url, timeout=REQUEST_TIMEOUT, proxy=proxy_for(url)
        ) as response:
            if response.status == 200:
                return _coerce_space_data(await response.json())
            else:
                return None
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return None


async def list_available_apps() -> list[AppInfo]:
    """List apps available on Hugging Face Spaces."""
    # Explicit proxy resolution (HTTP_PROXY/HTTPS_PROXY/NO_PROXY) —
    # deliberately NOT trust_env=True, which would also read ~/.netrc
    # and silently attach its credentials (see utils/proxy.py).
    async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
        # Fetch the list of authorized app IDs
        try:
            async with session.get(
                AUTHORIZED_APP_LIST_URL, proxy=proxy_for(AUTHORIZED_APP_LIST_URL)
            ) as response:
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
        async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
            data = await _fetch_all_spaces(session, token)
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
        logger.warning("Could not list HF Spaces: %s", exc)
        return []

    apps = []
    for item in data:
        app_info = _build_app_info(item)
        if app_info is not None:
            apps.append(app_info)

    return apps
