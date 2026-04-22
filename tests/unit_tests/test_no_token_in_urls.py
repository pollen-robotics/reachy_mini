"""Regression test: HF tokens must never appear in any URL string.

We authenticate to the central signaling server with an
``Authorization: Bearer <token>`` header. Passing the token as a
query parameter (``?token=...``) would leak it into HF Space access
logs, proxy logs, browser history, DevTools Network tab, and any
Referer header the user's browser might emit.

A prior version of the codebase built URLs of the form
``f"{self.central_uri}/events?token={self.hf_token}"``. This test
exists so that a future refactor can't silently re-introduce that
shape.

The check is a plain substring grep rather than AST-level analysis:
any string literal, f-string, or template-literal that contains the
substring ``token=`` in the token-sensitive files fails the test.
That is slightly conservative (it would flag unrelated query params
named ``token``), but the blast radius of a false positive is zero —
just name your param something else.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Files that construct outbound URLs to the central signaling server.
# Add new files here when another module starts talking to central.
TOKEN_SENSITIVE_FILES = [
    "src/reachy_mini/media/central_signaling_relay.py",
    "src/reachy_mini/daemon/app/routers/hf_auth.py",
    "js/reachy-mini.js",
]

# Substring patterns that indicate a token leaking into a URL. Keep
# these patterns conservative — the goal is an obvious fail fast when
# someone writes ``f"{uri}/send?token={token}"`` or similar, not a
# smart URL parser.
FORBIDDEN_PATTERNS = [
    re.compile(r"\?token="),   # f-string / template literal
    re.compile(r'"token"\s*:\s*token\b'),  # params={"token": token} in Python
]

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("relpath", TOKEN_SENSITIVE_FILES)
def test_no_token_in_urls(relpath: str) -> None:
    """Assert the file never embeds an HF token in a URL/querystring."""
    path = REPO_ROOT / relpath
    assert path.exists(), f"Expected file {path} to exist"

    source = path.read_text(encoding="utf-8")
    offending: list[tuple[int, str]] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        # Ignore lines that are purely comments explaining why we don't
        # do this (we do have such comments near the fix sites).
        stripped = line.lstrip()
        if stripped.startswith(("#", "//", "*")):
            continue
        for pat in FORBIDDEN_PATTERNS:
            if pat.search(line):
                offending.append((lineno, line.rstrip()))
                break

    assert not offending, (
        f"{relpath} contains token-in-URL patterns — tokens must travel "
        f"in the Authorization header, never in URLs/query strings. "
        f"Offending lines:\n"
        + "\n".join(f"  {lineno}: {text}" for lineno, text in offending)
    )
