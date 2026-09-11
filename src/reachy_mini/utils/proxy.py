"""Explicit HTTP(S) proxy resolution for aiohttp call sites.

Why not ``ClientSession(trust_env=True)``: that flag makes aiohttp honour
``HTTP_PROXY``/``HTTPS_PROXY``/``NO_PROXY`` — but it ALSO makes it read
``~/.netrc`` for HTTP auth, and aiohttp does not let netrc defer to an
explicit ``Authorization`` header: with a netrc entry matching the request
host (or a ``default`` entry, which pip/git tooling commonly writes on dev
machines), every request that carries a Bearer header raises
``ValueError("Cannot combine AUTHORIZATION header with AUTH argument")``,
and unauthenticated requests silently gain netrc Basic credentials. Every
authenticated Hugging Face / central-relay call in this package sends an
explicit Bearer header, so ``trust_env=True`` turns a stray developer
``.netrc`` into a hard failure of login, app update checks and remote
access.

Instead, this module resolves the proxy with the same stdlib machinery
``requests`` uses (``urllib.request.getproxies`` / ``proxy_bypass``) and
call sites pass it explicitly as aiohttp's per-request ``proxy=`` argument.
Environment semantics are identical to ``trust_env`` (scheme-keyed
``HTTP_PROXY``/``HTTPS_PROXY``, ``NO_PROXY`` bypass, https-scheme proxy
URLs skipped) with zero netrc involvement.

Guarded by ``tests/unit_tests/test_proxy_env_support.py``: an AST check
fails if a ``ClientSession`` is created with ``trust_env`` or if a session
request is made without ``proxy=``.
"""

from __future__ import annotations

import logging
from urllib.parse import urlsplit
from urllib.request import getproxies, proxy_bypass

logger = logging.getLogger(__name__)

# aiohttp's per-request `proxy=` only supports plain-HTTP proxy URLs (it
# tunnels HTTPS through them with CONNECT). `trust_env=True` silently skips
# https-scheme proxy URLs with a warning; we mirror that behaviour instead
# of letting aiohttp raise at request time. Warn once, not per request.
_warned_unsupported_proxy: set[str] = set()


def proxy_for(url: str) -> str | None:
    """Return the proxy URL to use for ``url``, or ``None`` to go direct.

    Reads ``HTTP_PROXY`` / ``HTTPS_PROXY`` (upper or lower case) via
    ``urllib.request.getproxies()`` and honours ``NO_PROXY`` via
    ``proxy_bypass()`` — the same stdlib machinery ``requests`` builds on.
    The lookup is strictly scheme-keyed, matching aiohttp's own
    ``trust_env`` behaviour: ``ALL_PROXY`` is NOT honoured (``requests``
    would fall back to it, aiohttp never did — configure the scheme
    variables explicitly). Resolved on every call so an environment change
    takes effect without recreating sessions, matching ``trust_env``
    semantics.

    Args:
        url: The absolute request URL.

    Returns:
        The proxy URL for the request's scheme, or ``None`` when no proxy
        is configured, the host matches ``NO_PROXY``, or the configured
        proxy URL uses a scheme aiohttp cannot tunnel through.

    """
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        # Malformed URL (e.g. bad IPv6 literal): let the HTTP client produce
        # its own error for it; resolving "no proxy" must never raise.
        return None
    if not host:
        return None
    try:
        # Honours NO_PROXY (and platform bypass rules on macOS/Windows).
        if proxy_bypass(host):
            return None
    except (TypeError, OSError):
        # A failed bypass lookup means "don't bypass" — the request still
        # goes through the proxy (and over TLS for every call site here).
        # Same exception set `requests` tolerates (gaierror is an OSError).
        pass
    proxy = getproxies().get(parts.scheme)
    if not proxy:
        return None
    proxy_scheme = urlsplit(proxy).scheme
    if proxy_scheme != "http":
        # Parity with aiohttp's trust_env, which skips non-HTTP proxy URLs
        # (per-request `proxy=` would raise ValueError on them instead).
        if proxy not in _warned_unsupported_proxy:
            _warned_unsupported_proxy.add(proxy)
            logger.warning(
                "Ignoring %s proxy %r: aiohttp only supports plain-HTTP "
                "proxies (HTTPS is tunnelled through them with CONNECT).",
                proxy_scheme or "schemeless",
                proxy,
            )
        return None
    return proxy
