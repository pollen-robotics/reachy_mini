"""Network-level request guards for the Reachy Mini daemon HTTP API.

The daemon API is unauthenticated. On wireless robots it must remain
reachable from the local network (the dashboard is opened from the user's
browser), which exposes state-changing endpoints (HF token management, app
install, ...) to two classes of remote attackers:

1. Web pages loaded in a browser on the LAN (cross-site request forgery and
   DNS rebinding): a page served from the internet can fire POST requests at
   ``http://reachy-mini.local:8000`` or at a hostname the attacker rebinds to
   the robot's IP.
2. Arbitrary hosts on the LAN talking to the API directly.

This module addresses class 1 by validating the ``Host`` and ``Origin``
headers of incoming requests:

- The ``Host`` header must be an IP literal, ``localhost``, or an mDNS
  ``.local`` name. A public DNS name pointing at the robot (DNS rebinding)
  is rejected.
- State-changing requests (anything but GET/HEAD/OPTIONS) that carry an
  ``Origin`` header must originate from the robot itself, from a
  ``localhost``/``.local`` origin, from a private-range IP origin, or from
  the desktop app's Tauri webview. Requests without an ``Origin`` header
  (curl, native SDK clients) are unaffected.

Class 2 requires real authentication (e.g. a pairing secret established
during provisioning) and is not solved here.
"""

import ipaddress
import re
from typing import Awaitable, Callable, Optional
from urllib.parse import urlsplit

from fastapi import FastAPI, Request, Response
from starlette.responses import PlainTextResponse

# Origins allowed to read API responses cross-origin (CORS). Mirrors the
# rules enforced by the request guard below: local hostnames, mDNS names,
# loopback/private IPv4 literals, and the desktop app's Tauri webview.
ALLOWED_ORIGIN_REGEX = (
    r"^(tauri://localhost"
    r"|https?://("
    r"localhost"
    r"|tauri\.localhost"
    r"|127\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|\[::1\]"
    r"|[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.local"
    r"|10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|169\.254\.\d{1,3}\.\d{1,3}"
    r")(:\d+)?)$"
)

_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_LOCAL_HOSTNAMES = {"localhost", "tauri.localhost"}
_MDNS_HOSTNAME_RE = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)*\.local$", re.IGNORECASE)


def _parse_ip(host: str) -> Optional[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _hostname_of(netloc: str) -> Optional[str]:
    """Extract the hostname (no port, no brackets) from a netloc string."""
    try:
        return urlsplit(f"//{netloc}").hostname
    except ValueError:
        return None


def _is_local_name(host: str) -> bool:
    return host in _LOCAL_HOSTNAMES or _MDNS_HOSTNAME_RE.match(host) is not None


def is_trusted_host(host: Optional[str]) -> bool:
    """Check the request ``Host`` header hostname.

    IP literals are trusted (direct LAN access by IP); DNS names are only
    trusted if they are local names, which defeats DNS rebinding (the
    attacker's rebound domain shows up here as a public DNS name).
    """
    if not host:
        return False
    if _parse_ip(host) is not None:
        return True
    return _is_local_name(host)


def is_trusted_origin(origin: str, request_host: Optional[str]) -> bool:
    """Check the ``Origin`` header of a state-changing request."""
    try:
        parts = urlsplit(origin)
    except ValueError:
        return False
    if parts.scheme == "tauri":
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


def add_local_network_guard(app: FastAPI) -> None:
    """Install the Host/Origin validation middleware on the app."""

    @app.middleware("http")
    async def local_network_guard(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        host = _hostname_of(request.headers.get("host", ""))
        if not is_trusted_host(host):
            return PlainTextResponse("Untrusted Host header.", status_code=400)

        if request.method not in _SAFE_METHODS:
            origin = request.headers.get("origin")
            if origin is not None and not is_trusted_origin(origin, host):
                return PlainTextResponse(
                    "Cross-origin request rejected.", status_code=403
                )

        return await call_next(request)
