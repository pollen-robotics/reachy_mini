"""The HF Spaces catalog listing: plain HTTP, only the fields the store reads."""

from typing import Any

import pytest
from aiohttp import web

from reachy_mini.apps.sources import hf_auth, hf_space

APP_STORE_FIELDS = {
    "author",
    "cardData",
    "createdAt",
    "lastModified",
    "likes",
    "private",
    "runtime",
    "sdk",
    "tags",
}


class _SpacesServer:
    """Local stand-in for https://huggingface.co/api/spaces."""

    def __init__(self, pages: list[list[dict[str, Any]]], status: int = 200) -> None:
        self.pages = pages
        self.status = status
        self.requests: list[web.Request] = []
        self.url = ""

    async def _handle(self, request: web.Request) -> web.Response:
        self.requests.append(request)
        if self.status != 200:
            return web.Response(status=self.status)
        page = int(request.query.get("page", "0"))
        headers = {}
        if page + 1 < len(self.pages):
            headers["Link"] = f'<{self.url}?page={page + 1}>; rel="next"'
        return web.json_response(self.pages[page], headers=headers)

    async def __aenter__(self) -> "_SpacesServer":
        app = web.Application()
        app.router.add_get("/api/spaces", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        self.url = f"http://127.0.0.1:{port}/api/spaces"
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._runner.cleanup()


def _space(space_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": space_id,
        "likes": 3,
        "cardData": {"short_description": f"desc of {space_id}"},
        "siblings": [{"rfilename": "app/__main__.py"}],
        **extra,
    }


@pytest.fixture
def _no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf_auth, "get_hf_token", lambda: None)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_token")
async def test_list_all_apps_requests_only_app_store_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _SpacesServer([[_space("owner/app")]]) as server:
        monkeypatch.setattr(hf_space, "HF_SPACES_API_URL", server.url)

        apps = await hf_space.list_all_apps()

    assert [a.name for a in apps] == ["app"]
    assert apps[0].description == "desc of owner/app"
    assert apps[0].extra["likes"] == 3
    assert "siblings" not in apps[0].extra

    query = server.requests[0].query
    assert query["filter"] == hf_space.HF_SPACES_FILTER
    assert "full" not in query
    assert set(query.getall("expand")) == APP_STORE_FIELDS
    assert "Authorization" not in server.requests[0].headers


@pytest.mark.asyncio
async def test_list_all_apps_sends_bearer_token_when_authenticated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hf_auth, "get_hf_token", lambda: "hf_secret")
    async with _SpacesServer([[_space("owner/private-app", private=True)]]) as server:
        monkeypatch.setattr(hf_space, "HF_SPACES_API_URL", server.url)

        apps = await hf_space.list_all_apps()

    assert [a.name for a in apps] == ["private-app"]
    assert server.requests[0].headers["Authorization"] == "Bearer hf_secret"


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_token")
async def test_list_all_apps_follows_next_page_links(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = [[_space("owner/one")], [_space("owner/two")]]
    async with _SpacesServer(pages) as server:
        monkeypatch.setattr(hf_space, "HF_SPACES_API_URL", server.url)

        apps = await hf_space.list_all_apps()

    assert [a.name for a in apps] == ["one", "two"]
    assert len(server.requests) == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_token")
async def test_list_all_apps_returns_empty_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _SpacesServer([[_space("owner/app")]], status=500) as server:
        monkeypatch.setattr(hf_space, "HF_SPACES_API_URL", server.url)

        apps = await hf_space.list_all_apps()

    assert apps == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_token")
async def test_list_all_apps_returns_empty_when_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Port 9 (discard) refuses connections on loopback.
    monkeypatch.setattr(hf_space, "HF_SPACES_API_URL", "http://127.0.0.1:9/api/spaces")

    apps = await hf_space.list_all_apps()

    assert apps == []
