"""Custom ASGI middleware for the daemon HTTP app."""

import ipaddress
import json
import logging
import re
from collections.abc import Iterable
from urllib.parse import urlsplit

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)


class _BodyTooLarge(Exception):
    """Raised internally once a request body exceeds the configured limit."""


class MaxBodySizeMiddleware:
    """Reject requests to *paths* whose body exceeds *max_body_size* bytes.

    The limit is enforced *before* the body is parsed, so a large upload is
    never read in full:

    - an explicit ``Content-Length`` over the limit is rejected outright,
      before a single byte of the body is read;
    - a body that crosses the limit while streaming (chunked transfer, or an
      understated/absent ``Content-Length``) is aborted as soon as the
      threshold is passed.

    Other paths and non-HTTP scopes are passed through untouched.
    """

    def __init__(
        self, app: ASGIApp, *, max_body_size: int, paths: Iterable[str]
    ) -> None:
        """Wrap *app*, capping bodies on *paths* at *max_body_size* bytes."""
        self.app = app
        self.max_body_size = max_body_size
        self.paths = frozenset(paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Enforce the body-size limit on matching requests."""
        if scope["type"] != "http" or scope.get("path") not in self.paths:
            await self.app(scope, receive, send)
            return

        # Fast path: an honest, oversized Content-Length is rejected before the
        # body is read at all (covers curl, browsers, requests, ...).
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    break
                if declared > self.max_body_size:
                    await self._send_too_large(send)
                    return
                break

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_size:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            # The body parser raises before producing a response, so the
            # response stream is still ours to write.
            if not response_started:
                await self._send_too_large(send)
            else:
                logger.warning(
                    "Request body exceeded %d bytes after the response started",
                    self.max_body_size,
                )

    async def _send_too_large(self, send: Send) -> None:
        body = json.dumps(
            {"detail": f"Request body too large; maximum is {self.max_body_size} bytes"}
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


# ---------------------------------------------------------------------------
# Local-network request guard (CAN-2026-2032024)
# ---------------------------------------------------------------------------

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_LOCAL_HOSTNAMES = frozenset({"localhost", "tauri.localhost"})
_WEBVIEW_SCHEMES = frozenset({"tauri", "capacitor"})
_MDNS_HOSTNAME_RE = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)*\.local$", re.IGNORECASE)


def _parse_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _hostname_of(netloc: str) -> str | None:
    """Extract the hostname (no port, no brackets) from a netloc string."""
    try:
        return urlsplit(f"//{netloc}").hostname
    except ValueError:
        return None


def _is_local_name(host: str) -> bool:
    return host in _LOCAL_HOSTNAMES or _MDNS_HOSTNAME_RE.match(host) is not None


def is_trusted_host(host: str | None) -> bool:
    """Check a request ``Host`` header hostname.

    IP literals are trusted (direct LAN access by IP); DNS names only if they
    are local names. A DNS-rebinding attack necessarily arrives under the
    attacker's public DNS name, which is neither.
    """
    if not host:
        return False
    if _parse_ip(host) is not None:
        return True
    return _is_local_name(host)


def is_trusted_origin(origin: str, request_host: str | None) -> bool:
    """Check the ``Origin`` header of a state-changing request."""
    try:
        parts = urlsplit(origin)
    except ValueError:
        return False
    if parts.scheme in _WEBVIEW_SCHEMES:
        return True
    if parts.scheme not in ("http", "https"):
        return False
    host = parts.hostname
    if not host:
        return False
    if request_host and host == request_host:
        return True
    if _is_local_name(host):
        return True
    ip = _parse_ip(host)
    if ip is not None:
        return ip.is_loopback or ip.is_private or ip.is_link_local
    return False


class LocalNetworkGuardMiddleware:
    """Reject browser-borne cross-site attacks on the unauthenticated API.

    The API relies on network locality instead of authentication. Two browser
    tricks let an internet page reach it anyway (CAN-2026-2032024):

    - DNS rebinding: the attacker's domain re-resolves to the robot's LAN IP,
      making the page same-origin with the API. Blocked by requiring the
      ``Host`` header to be an IP literal, ``localhost``, or a ``.local``
      mDNS name -- a rebound request arrives under a public DNS name.
    - Cross-site request forgery: CORS (see ``CORS_ORIGIN_REGEX``) only stops
      the page from *reading* responses; a preflight-free "simple" POST still
      executes server-side. Blocked by rejecting state-changing requests
      whose ``Origin`` is not the robot itself, a local/private-network
      origin, or a native webview scheme (Tauri/Capacitor).

    Requests without an ``Origin`` header (curl, the Python SDK, native apps)
    pass untouched: this guards against browsers acting as confused deputies,
    not against direct LAN clients.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap *app* with Host and Origin validation."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Validate the Host and Origin headers of HTTP requests."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        host_header = ""
        origin: str | None = None
        for name, value in scope.get("headers", []):
            if name == b"host" and not host_header:
                host_header = value.decode("latin-1")
            elif name == b"origin" and origin is None:
                origin = value.decode("latin-1")

        host = _hostname_of(host_header)
        if not is_trusted_host(host):
            await self._reject(send, 400, "Untrusted Host header")
            return

        method = str(scope.get("method", "GET")).upper()
        if (
            method not in _SAFE_METHODS
            and origin is not None
            and not is_trusted_origin(origin, host)
        ):
            await self._reject(send, 403, "Cross-origin request rejected")
            return

        await self.app(scope, receive, send)

    async def _reject(self, send: Send, status: int, detail: str) -> None:
        body = json.dumps({"detail": detail}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
