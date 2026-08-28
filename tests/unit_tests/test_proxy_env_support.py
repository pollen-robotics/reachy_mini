"""Regression tests for HTTP(S) proxy support on our aiohttp call sites.

`requests` (and therefore `huggingface_hub`) reads HTTP_PROXY/HTTPS_PROXY
from the environment by default, but `aiohttp.ClientSession` does not. We
deliberately do NOT use aiohttp's `trust_env=True` for this: that flag also
reads `~/.netrc`, and aiohttp refuses requests that combine a netrc match
with an explicit `Authorization` header (`ValueError: Cannot combine
AUTHORIZATION header with AUTH argument`), which would break every
authenticated HF/central call on machines with a developer `.netrc`.
Instead, every aiohttp request passes an explicit
`proxy=reachy_mini.utils.proxy.proxy_for(url)`.

The end-to-end tests run a minimal HTTP server that doubles as an origin
server (origin-form request line) and as a forward proxy (absolute-form
request line), so they assert on the request line actually received. The
netrc tests pin the regression that motivated the explicit-proxy design.
"""

import ast
import asyncio
from pathlib import Path

import pytest

from reachy_mini.apps.sources import app_update_checker, hf_space
from reachy_mini.utils import proxy as proxy_mod
from reachy_mini.utils.proxy import proxy_for

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "reachy_mini"


class RecordingServer:
    """Minimal HTTP/1.1 server recording request line + headers of every request."""

    def __init__(self, body: str) -> None:
        """Serve `body` as a JSON response to any request."""
        self.body = body.encode()
        self.request_lines: list[str] = []
        self.header_blocks: list[list[str]] = []
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

    def all_headers(self) -> list[str]:
        """Every header line received, lowercased, across all requests."""
        return [h.lower() for block in self.header_blocks for h in block]

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        request_line = (await reader.readline()).decode().strip()
        self.request_lines.append(request_line)
        headers: list[str] = []
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            headers.append(line.decode().strip())
        self.header_blocks.append(headers)
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
    """Drop any ambient proxy/netrc configuration so tests are deterministic."""
    for name in ("http_proxy", "https_proxy", "no_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.delenv("NETRC", raising=False)


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


# ---- Regression: a developer ~/.netrc must never break or taint our calls
#
# This is why we resolve proxies explicitly instead of trust_env=True: with
# trust_env, aiohttp reads netrc and (a) raises ValueError on any request
# carrying an explicit Authorization header when the netrc matches, and
# (b) silently attaches netrc Basic credentials to unauthenticated requests.


@pytest.fixture
def _netrc_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    """Point NETRC at a file with a catch-all `default` entry."""
    netrc_file = tmp_path / "netrc"
    netrc_file.write_text("default login devuser password devpass\n")
    netrc_file.chmod(0o600)
    monkeypatch.setenv("NETRC", str(netrc_file))
    return netrc_file


@pytest.mark.asyncio
@pytest.mark.usefixtures("_clean_proxy_env", "_netrc_default")
async def test_authorized_request_survives_netrc_direct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Bearer-authenticated call works with a netrc present (no proxy).

    With trust_env=True this raised `ValueError: Cannot combine
    AUTHORIZATION header with AUTH argument` before the request was sent.
    """
    body = '{"sha": "abc123", "lastModified": "2026-01-01"}'
    async with RecordingServer(body) as origin:
        monkeypatch.setattr(
            app_update_checker,
            "HF_SPACES_API_URL",
            f"http://127.0.0.1:{origin.port}/api/spaces",
        )

        result = await app_update_checker.get_space_latest_sha(
            "owner/app", token="hf_test_token"
        )

    assert result == ("abc123", "2026-01-01")
    headers = origin.all_headers()
    assert "authorization: bearer hf_test_token" in headers
    assert not any(h.startswith("authorization: basic") for h in headers)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_clean_proxy_env", "_netrc_default")
async def test_authorized_request_survives_netrc_via_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Bearer-authenticated call works with netrc present AND a proxy set."""
    body = '{"sha": "abc123", "lastModified": "2026-01-01"}'
    async with RecordingServer(body) as proxy:
        monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.port}")
        monkeypatch.setattr(
            app_update_checker, "HF_SPACES_API_URL", "http://hf.invalid/api/spaces"
        )

        result = await app_update_checker.get_space_latest_sha(
            "owner/app", token="hf_test_token"
        )

    assert result == ("abc123", "2026-01-01")
    assert proxy.request_lines == [
        "GET http://hf.invalid/api/spaces/owner/app HTTP/1.1"
    ]
    assert "authorization: bearer hf_test_token" in proxy.all_headers()


@pytest.mark.asyncio
@pytest.mark.usefixtures("_clean_proxy_env", "_netrc_default")
async def test_unauthenticated_request_gains_no_netrc_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A netrc entry is never silently attached to an unauthenticated call."""
    async with RecordingServer("[]") as origin:
        monkeypatch.setattr(
            hf_space,
            "AUTHORIZED_APP_LIST_URL",
            f"http://127.0.0.1:{origin.port}/app-list.json",
        )

        apps = await hf_space.list_available_apps()

    assert apps == []
    assert not any(
        h.startswith("authorization:") for h in origin.all_headers()
    ), "netrc credentials must never be attached implicitly"


# ---- Unit: proxy_for environment semantics


@pytest.mark.usefixtures("_clean_proxy_env")
def test_proxy_for_returns_none_without_configuration() -> None:
    """No proxy env means direct connection."""
    assert proxy_for("https://huggingface.co/api") is None
    assert proxy_for("http://example.com/x") is None


@pytest.mark.usefixtures("_clean_proxy_env")
def test_proxy_for_is_scheme_keyed(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP_PROXY applies to http URLs only; HTTPS_PROXY to https URLs."""
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.local:3128")
    assert proxy_for("http://example.com/x") == "http://proxy.local:3128"
    assert proxy_for("https://example.com/x") is None
    monkeypatch.setenv("HTTPS_PROXY", "http://sproxy.local:3129")
    assert proxy_for("https://example.com/x") == "http://sproxy.local:3129"


@pytest.mark.usefixtures("_clean_proxy_env")
def test_proxy_for_honours_no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """A NO_PROXY host bypasses the configured proxy."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.local:3128")
    monkeypatch.setenv("NO_PROXY", "huggingface.co")
    assert proxy_for("https://huggingface.co/api") is None
    assert proxy_for("https://elsewhere.example/api") == "http://proxy.local:3128"


@pytest.mark.usefixtures("_clean_proxy_env")
def test_proxy_for_skips_non_http_proxy_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An https:// proxy URL is skipped (aiohttp can't tunnel through it).

    Mirrors aiohttp's own trust_env behaviour, which skips such entries with
    a warning; passing them per-request would raise ValueError instead.
    """
    monkeypatch.setenv("HTTPS_PROXY", "https://secure-proxy.local:3128")
    assert proxy_for("https://huggingface.co/api") is None


@pytest.mark.usefixtures("_clean_proxy_env")
def test_proxy_for_handles_hostless_urls() -> None:
    """A URL without a host resolves to a direct connection, not a crash."""
    assert proxy_for("not a url") is None
    assert proxy_for("") is None


# ---- Guards: no trust_env, and every session request passes proxy=

REQUEST_METHODS = {
    "get",
    "post",
    "put",
    "delete",
    "patch",
    "head",
    "options",
    "request",
    "ws_connect",
}


def _is_client_session_ctor(node: ast.expr) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    return name == "ClientSession"


def _annotation_is_client_session(ann: ast.expr | None) -> bool:
    """True for `aiohttp.ClientSession`, `ClientSession`, or quoted forms."""
    if ann is None:
        return False
    if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
        return "ClientSession" in ann.value
    name = ann.attr if isinstance(ann, ast.Attribute) else getattr(ann, "id", "")
    return name == "ClientSession"


def _proxy_kwarg_is_valid(call: ast.Call, url_arg_index: int) -> bool:
    """True iff `proxy=proxy_for(<same URL expression as the request>)`.

    `proxy=None` (or any other value) fully restores the "proxy ignored"
    bug, and `proxy=proxy_for(other_url)` resolves against the wrong host,
    so both are rejected: the kwarg must be a `proxy_for(...)` call whose
    argument is textually identical to the request's URL argument.
    """
    proxy_kw = next((kw.value for kw in call.keywords if kw.arg == "proxy"), None)
    if not isinstance(proxy_kw, ast.Call):
        return False
    func = proxy_kw.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    if name != "proxy_for" or len(proxy_kw.args) != 1:
        return False
    if len(call.args) <= url_arg_index:
        return False
    return ast.dump(proxy_kw.args[0]) == ast.dump(call.args[url_arg_index])


def _guard_module(path: Path) -> tuple[list[int], list[int]]:
    """Return (ctor offenders, request-calls-without-valid-proxy offenders)."""
    tree = ast.parse(path.read_text(), filename=str(path))

    ctor_offenders: list[int] = []
    session_names: set[str] = set()  # `... as session` / `x = ClientSession()`
    session_attrs: set[str] = set()  # `self._http = ClientSession()`

    for node in ast.walk(tree):
        # 1. constructors: no trust_env and no opaque **kwargs; collect names
        if isinstance(node, ast.Call) and _is_client_session_ctor(node):
            if any(kw.arg in ("trust_env", None) for kw in node.keywords):
                ctor_offenders.append(node.lineno)
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if _is_client_session_ctor(item.context_expr) and isinstance(
                    item.optional_vars, ast.Name
                ):
                    session_names.add(item.optional_vars.id)
        if isinstance(node, ast.Assign) and _is_client_session_ctor(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    session_names.add(target.id)
                elif isinstance(target, ast.Attribute):
                    session_attrs.add(target.attr)
        if isinstance(node, ast.AnnAssign) and (
            (node.value is not None and _is_client_session_ctor(node.value))
            or _annotation_is_client_session(node.annotation)
        ):
            if isinstance(node.target, ast.Name):
                session_names.add(node.target.id)
            elif isinstance(node.target, ast.Attribute):
                session_attrs.add(node.target.attr)

    # Sessions received as function parameters annotated ClientSession
    # (e.g. `_fetch_space_data(session: aiohttp.ClientSession, ...)`).
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in [*node.args.args, *node.args.kwonlyargs]:
                if _annotation_is_client_session(arg.annotation):
                    session_names.add(arg.arg)

    bad_requests: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in REQUEST_METHODS):
            continue
        recv = func.value
        is_session = (
            (isinstance(recv, ast.Name) and recv.id in session_names)
            or (isinstance(recv, ast.Attribute) and recv.attr in session_attrs)
            # session-less module helper: aiohttp.request("GET", url)
            or (isinstance(recv, ast.Name) and recv.id == "aiohttp")
        )
        if not is_session:
            continue
        # `.request()` / `.ws_connect()` take (method, url); the verb
        # helpers take (url, ...).
        url_arg_index = 1 if func.attr == "request" else 0
        if not _proxy_kwarg_is_valid(node, url_arg_index):
            bad_requests.append(node.lineno)

    return ctor_offenders, bad_requests


def test_no_aiohttp_session_uses_trust_env() -> None:
    """No ClientSession may pass trust_env (or hide it in **kwargs)."""
    assert SRC_ROOT.is_dir(), f"source root missing: {SRC_ROOT}"
    offenders = {
        str(path.relative_to(SRC_ROOT)): lines
        for path in sorted(SRC_ROOT.rglob("*.py"))
        if (lines := _guard_module(path)[0])
    }
    assert offenders == {}, (
        "trust_env on aiohttp.ClientSession also reads ~/.netrc and breaks "
        "Authorization-header requests; resolve the proxy explicitly with "
        f"reachy_mini.utils.proxy.proxy_for instead. Offenders: {offenders}"
    )


def test_every_session_request_passes_an_explicit_proxy() -> None:
    """Every aiohttp request must pass proxy=proxy_for(<its own URL>).

    `proxy=None` or `proxy=proxy_for(<a different URL>)` are rejected too:
    the first silently restores the ignored-proxy bug, the second resolves
    NO_PROXY against the wrong host.
    """
    assert SRC_ROOT.is_dir(), f"source root missing: {SRC_ROOT}"
    offenders = {
        str(path.relative_to(SRC_ROOT)): lines
        for path in sorted(SRC_ROOT.rglob("*.py"))
        if (lines := _guard_module(path)[1])
    }
    assert offenders == {}, (
        "aiohttp requests ignore HTTP_PROXY/HTTPS_PROXY unless given an "
        "explicit proxy=proxy_for(url) with the request's own URL (see "
        f"utils/proxy.py). Offenders: {offenders}"
    )
