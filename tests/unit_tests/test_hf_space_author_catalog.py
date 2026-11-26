"""Tests for author-scoped HF Spaces catalog merge (issue #1374)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from huggingface_hub import SpaceInfo

from reachy_mini.apps.sources import hf_space


def _space(space_id: str, likes: int = 0, **extra: object) -> SimpleNamespace:
    return SimpleNamespace(id=space_id, likes=likes, cardData={}, **extra)


@pytest.fixture
def catalog(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, AsyncMock]:
    """Mock both providers while exercising the production catalog entry point."""
    api = MagicMock()
    api.whoami.return_value = {"name": "me"}
    api.list_spaces.return_value = []
    public = AsyncMock(return_value=[{"id": "org/popular", "likes": 9}])
    monkeypatch.setattr(hf_space, "HfApi", lambda: api)
    monkeypatch.setattr(hf_space, "_fetch_all_spaces", public)
    monkeypatch.setattr(hf_space.hf_auth, "get_hf_token", lambda: "offline-fixture")
    return api, public


def test_merge_space_payloads_keeps_first_id() -> None:
    """First occurrence of a Space id wins during merge."""
    first = {"id": "user/app", "likes": 1}
    second = {"id": "user/app", "likes": 99}
    other = {"id": "org/other", "likes": 2}
    merged = hf_space._merge_space_payloads([first], [second, other])
    assert merged == [first, other]


@pytest.mark.asyncio
@pytest.mark.parametrize("token", [None, False])
async def test_list_all_spaces_without_token_skips_author_query(
    monkeypatch: pytest.MonkeyPatch,
    catalog: tuple[MagicMock, AsyncMock],
    token: None | bool,
) -> None:
    """Anonymous requests preserve the False sentinel and skip author discovery."""
    api, public = catalog
    monkeypatch.setattr(hf_space.hf_auth, "get_hf_token", lambda: token)
    apps = await hf_space.list_all_apps()
    assert [a.extra["id"] for a in apps] == ["org/popular"]
    public.assert_awaited_once()
    assert public.await_args.args[1] is False
    api.list_spaces.assert_not_called()
    api.whoami.assert_not_called()


@pytest.mark.asyncio
async def test_list_all_spaces_merges_author_spaces(
    catalog: tuple[MagicMock, AsyncMock],
) -> None:
    """Unlimited author results win collisions and include private, zero-like apps."""
    api, public = catalog
    public.return_value = [
        {"id": "me/also-public", "likes": 5},
        {"id": "org/popular", "likes": 9},
    ]
    api.list_spaces.return_value = iter(
        [_space("me/private", private=True), _space("me/also-public")]
    )
    apps = await hf_space.list_all_apps()
    assert [a.extra["id"] for a in apps] == [
        "me/private",
        "me/also-public",
        "org/popular",
    ]
    assert apps[0].extra["private"] is True
    assert apps[0].extra["likes"] == 0
    assert apps[1].extra["likes"] == 0
    assert public.await_args.args[1] == "offline-fixture"
    api.whoami.assert_called_once_with(token="offline-fixture")
    api.list_spaces.assert_called_once_with(
        filter=hf_space.HF_SPACES_FILTER,
        author="me",
        full=True,
        token="offline-fixture",
    )


@pytest.mark.asyncio
async def test_list_all_spaces_survives_whoami_failure(
    catalog: tuple[MagicMock, AsyncMock],
) -> None:
    """A whoami failure must not drop the public catalog."""
    api, _ = catalog
    api.whoami.side_effect = RuntimeError("offline")
    apps = await hf_space.list_all_apps()
    assert [a.extra["id"] for a in apps] == ["org/popular"]
    api.whoami.assert_called_once_with(token="offline-fixture")
    api.list_spaces.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("page", [1, 2])
async def test_author_pagination_failure_preserves_public_catalog(
    catalog: tuple[MagicMock, AsyncMock], caplog: pytest.LogCaptureFixture, page: int
) -> None:
    """Lazy failures at either page must fall back to the complete public result."""
    api, public = catalog
    public.return_value = [{"id": "org/popular"}, {"id": "org/second"}]

    def list_spaces(**kwargs: object) -> Iterator[SimpleNamespace]:
        if page == 2:
            yield _space("me/partial-private")
        raise RuntimeError(f"author page {page} failed")

    api.list_spaces.side_effect = list_spaces
    apps = await hf_space.list_all_apps()
    assert [a.extra["id"] for a in apps] == ["org/popular", "org/second"]
    assert f"author page {page} failed" in caplog.text


@pytest.mark.asyncio
async def test_author_request_failure_preserves_public_catalog(
    catalog: tuple[MagicMock, AsyncMock], caplog: pytest.LogCaptureFixture
) -> None:
    """An eager request failure must also leave the public result intact."""
    api, _ = catalog
    api.list_spaces.side_effect = RuntimeError("request failed")
    apps = await hf_space.list_all_apps()
    assert [a.extra["id"] for a in apps] == ["org/popular"]
    assert "request failed" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", [None, {}, {"name": ""}, {"name": 42}])
async def test_missing_author_preserves_public_catalog(
    catalog: tuple[MagicMock, AsyncMock], identity: object
) -> None:
    """Malformed identities cannot turn the author query into an unbounded catalog."""
    api, _ = catalog
    api.whoami.return_value = identity
    apps = await hf_space.list_all_apps()
    assert [a.extra["id"] for a in apps] == ["org/popular"]
    api.whoami.assert_called_once_with(token="offline-fixture")
    api.list_spaces.assert_not_called()


def test_author_space_info_is_normalized() -> None:
    """Real Hub objects retain card metadata and dates without shipping file lists."""
    created = datetime(2026, 10, 1, tzinfo=timezone.utc)
    info = SpaceInfo(
        id="me/private",
        private=True,
        likes=0,
        createdAt="2026-10-01T00:00:00.000Z",
        cardData={"short_description": "Private app"},
        siblings=[{"rfilename": "app.py"}],
    )
    payloads = hf_space._space_payloads_from_hf_objects(iter([info, {}, None]))
    assert len(payloads) == 1
    assert payloads[0]["createdAt"] == created.isoformat()
    assert "siblings" not in payloads[0]
    app = hf_space._build_app_info(payloads[0])
    assert app is not None and app.description == "Private app"
    assert app.extra["private"] is True


@pytest.mark.asyncio
async def test_author_listing_is_unlimited_and_consumed_off_event_loop(
    catalog: tuple[MagicMock, AsyncMock],
) -> None:
    """Even authors with more than the public cap keep every app without blocking."""
    import threading

    api, _ = catalog
    event_loop_thread = threading.get_ident()
    count = hf_space.HF_SPACES_LIMIT + 1

    def list_spaces(**kwargs: object) -> Iterator[SimpleNamespace]:
        assert threading.get_ident() != event_loop_thread
        assert "limit" not in kwargs
        for i in range(count):
            yield _space(f"me/app-{i}")

    api.list_spaces.side_effect = list_spaces
    apps = await hf_space.list_all_apps()
    assert [a.extra["id"] for a in apps] == [
        *[f"me/app-{i}" for i in range(count)],
        "org/popular",
    ]
