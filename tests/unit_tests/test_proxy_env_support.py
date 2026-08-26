"""Regression tests for HTTP(S) proxy support on our aiohttp call sites.

`requests` (and therefore `huggingface_hub`) reads HTTP_PROXY/HTTPS_PROXY
from the environment by default, but `aiohttp.ClientSession` does not: it
only does so when built with `trust_env=True`. Every session we open must
set it, otherwise the Hugging Face calls made with aiohttp bypass a proxy
that the rest of the stack honours.

The end-to-end tests run a minimal HTTP server that doubles as an origin
server (origin-form request line) and as a forward proxy (absolute-form
request line), so they assert on the request line actually received.
"""

import ast
import asyncio
from pathlib import Path

import pytest

from reachy_mini.apps.sources import app_update_checker, hf_space

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "reachy_mini"


class RecordingServer:
    """Minimal HTTP/1.1 server recording the request line of every request."""

    def __init__(self, body: str) -> None:
        """Serve `body` as a JSON response to any request."""
        self.body = body.encode()
        self.request_lines: list[str] = []
        self._server: asyncio.AbstractServer | None = None

    @property
    def port(self) -> int:
        """Port the server is listening on."""
        assert self._server is not None
        return self._server.sockets[0].getsockname()[1]

    async def __aenter__(self) -> "RecordingServer":
        """Start listening on a loopback ephemeral port."""
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Stop listening and wait for the socket to close."""
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        request_line = (await reader.readline()).decode().strip()
        self.request_lines.append(request_line)
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: %d\r\n"
            b"Connection: close\r\n\r\n" % len(self.body)
        )
        writer.write(self.body)
        await writer.drain()
        writer.close()


@pytest.fixture
def _clean_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop any ambient proxy configuration so tests are deterministic."""
    for name in ("http_proxy", "https_proxy", "no_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)


# ---- End-to-end: requests go through the configured proxy


@pytest.mark.asyncio
@pytest.mark.usefixtures("_clean_proxy_env")
async def test_list_available_apps_uses_http_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The app-store listing reaches an unresolvable host via HTTP_PROXY."""
    async with RecordingServer("[]") as proxy:
        monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.port}")
        monkeypatch.setattr(
            hf_space, "AUTHORIZED_APP_LIST_URL", "http://hf.invalid/app-list.json"
        )

        apps = await hf_space.list_available_apps()

    assert apps == []
    assert proxy.request_lines == ["GET http://hf.invalid/app-list.json HTTP/1.1"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("_clean_proxy_env")
async def test_space_latest_sha_uses_http_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The app update check reaches the spaces API via HTTP_PROXY."""
    body = '{"sha": "abc123", "lastModified": "2026-01-01"}'
    async with RecordingServer(body) as proxy:
        monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.port}")
        monkeypatch.setattr(
            app_update_checker, "HF_SPACES_API_URL", "http://hf.invalid/api/spaces"
        )

        result = await app_update_checker.get_space_latest_sha("owner/app")

    assert result == ("abc123", "2026-01-01")
    assert proxy.request_lines == [
        "GET http://hf.invalid/api/spaces/owner/app HTTP/1.1"
    ]


@pytest.mark.asyncio
@pytest.mark.usefixtures("_clean_proxy_env")
async def test_no_proxy_configured_connects_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no proxy in the environment, the request goes straight out."""
    body = '{"sha": "def456", "lastModified": "2026-01-02"}'
    async with RecordingServer(body) as origin:
        monkeypatch.setattr(
            app_update_checker,
            "HF_SPACES_API_URL",
            f"http://127.0.0.1:{origin.port}/api/spaces",
        )

        result = await app_update_checker.get_space_latest_sha("owner/app")

    assert result == ("def456", "2026-01-02")
    assert origin.request_lines == ["GET /api/spaces/owner/app HTTP/1.1"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("_clean_proxy_env")
async def test_no_proxy_env_var_bypasses_the_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host listed in NO_PROXY is contacted directly, proxy notwithstanding."""
    async with RecordingServer("[]") as origin:
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")  # would refuse
        monkeypatch.setenv("NO_PROXY", "127.0.0.1")
        monkeypatch.setattr(
            hf_space,
            "AUTHORIZED_APP_LIST_URL",
            f"http://127.0.0.1:{origin.port}/app-list.json",
        )

        apps = await hf_space.list_available_apps()

    assert apps == []
    assert origin.request_lines == ["GET /app-list.json HTTP/1.1"]


# ---- Guard: no session may be opened without trust_env


def _sessions_without_trust_env(path: Path) -> list[int]:
    """Return the line numbers of ClientSession() calls missing trust_env=True."""
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "ClientSession":
            continue
        trust_env = next(
            (kw.value for kw in node.keywords if kw.arg == "trust_env"), None
        )
        if not (isinstance(trust_env, ast.Constant) and trust_env.value is True):
            offenders.append(node.lineno)
    return offenders


def test_every_aiohttp_session_trusts_the_environment() -> None:
    """Every aiohttp session in the package passes trust_env=True."""
    offenders = {
        str(path.relative_to(SRC_ROOT)): lines
        for path in sorted(SRC_ROOT.rglob("*.py"))
        if (lines := _sessions_without_trust_env(path))
    }
    assert offenders == {}, (
        "aiohttp.ClientSession without trust_env=True ignores HTTP_PROXY/"
        f"HTTPS_PROXY; add it at: {offenders}"
    )
