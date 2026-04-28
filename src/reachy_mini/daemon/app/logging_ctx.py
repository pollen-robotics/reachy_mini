"""Structured logging context for the Reachy Mini daemon.

Mirrors the mobile app's `src/logger/` layer (PR-A) so that a single
``X-Trace-Id`` header travels from the phone, through the optional
WebRTC ``http_proxy`` tunnel, into the daemon's HTTP handlers, and ends
up tagged on every Python log line emitted while servicing that
request.

What it provides
----------------
* :data:`trace_id_var` - a :class:`contextvars.ContextVar` holding the
  current trace id (4-char base36 to match the mobile side). Async-safe
  by design: every coroutine spawned inside a request copies its own
  copy, so concurrent requests do not bleed ids.
* :class:`TraceIdFilter` - a :class:`logging.Filter` that injects
  ``record.trace_id`` from the ContextVar so format strings can use
  ``%(trace_id)s``. It also exposes a sentinel ``"----"`` when no
  trace is active, making formatter strings unconditional.
* :func:`kv` / :func:`kv_log` - small helpers to emit structured
  ``event_name key=value key=value`` log lines without standing up a
  full structlog dependency.
* :func:`redact_dict` - walks an arbitrary dict and short-fingerprints
  any value whose *key* matches a sensitive pattern (token, bearer,
  password, ...). Use it on anything that crosses a log boundary.

Design notes
------------
* We don't replace existing ``logger.info(...)`` calls. The kv helpers
  are additive, available at strategic observation points (hf_auth,
  signaling relay, wifi router) without forcing a global rewrite.
* Trace ids are deliberately short (4 chars). Cardinality across one
  daemon session is low (typically a few hundred connection attempts);
  4 base36 chars give us 1.6M values, more than enough to dedupe.
* The redaction list is keyed *by name*, not *by value* - we don't
  scan strings looking for tokens, because that's both slow and
  unreliable. Instead, callers must pass tokens under a known key
  (``token=``, ``hf_token=``, ``authorization=``, ...) so the filter
  catches them.
"""

from __future__ import annotations

import contextvars
import logging
import re
import secrets
from typing import Any, Mapping

# 4-char base36 trace-id matches the mobile-side `newTraceId()` length.
# Async-safe via ContextVar: spawning a task COPIES the current context,
# so a request's trace stays scoped to that request.
trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "trace_id", default=None
)

_TRACE_PLACEHOLDER = "----"


def get_trace_id() -> str | None:
    """Return the current trace id, or ``None`` if outside a request."""
    return trace_id_var.get()


def set_trace_id(trace: str | None) -> contextvars.Token[str | None]:
    """Set the trace id for the current context.

    Returns a token that callers MUST pass back to :func:`reset_trace_id`
    once the work is done, to avoid leaking the trace across unrelated
    coroutines (FastAPI background tasks, tests, ...). Most consumers
    should use the :class:`TraceIdMiddleware` rather than calling this
    directly.
    """
    return trace_id_var.set(trace)


def reset_trace_id(token: contextvars.Token[str | None]) -> None:
    """Restore the trace id that was active before :func:`set_trace_id`."""
    trace_id_var.reset(token)


def new_trace_id() -> str:
    """Mint a fresh 4-char base36 trace id.

    ``secrets.randbits`` over ``random.random()``: the trace id is not
    a security boundary, but using ``secrets`` avoids depending on the
    global PRNG state which may be seeded from sources we don't control
    in CI / fork-after-init scenarios.
    """
    n = secrets.randbits(32)
    # base36 encode, pad/truncate to 4 chars.
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    while n > 0 and len(out) < 6:
        out = digits[n % 36] + out
        n //= 36
    return (out or "0000")[-4:].rjust(4, "0")


class TraceIdFilter(logging.Filter):
    """Inject the current trace id into every :class:`LogRecord`.

    Install it on each handler whose formatter uses ``%(trace_id)s``.
    When no trace is set we emit ``"----"`` (a stable 4-char filler)
    so the column width stays predictable in stderr logs.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D102, D401
        record.trace_id = trace_id_var.get() or _TRACE_PLACEHOLDER
        return True


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

_SENSITIVE_KEY_RE = re.compile(
    r"^(token|hf_token|hftoken|access_token|refresh_token|"
    r"authorization|bearer|password|secret|api_?key)$",
    re.IGNORECASE,
)


def redact_token(value: str) -> str:
    """Short-fingerprint a sensitive string so logs stay diff-able.

    Keeps 4 + 4 chars with an ellipsis in between for values long
    enough that a fingerprint is unambiguous; emits ``"REDACTED"`` for
    short ones.
    """
    if not value:
        return value
    if len(value) <= 12:
        return "REDACTED"
    return f"{value[:4]}…{value[-4:]}"


def redact_dict(data: Any, depth: int = 0) -> Any:
    """Recursively redact sensitive values in ``data``.

    Keys are matched against :data:`_SENSITIVE_KEY_RE`. Non-string
    values for sensitive keys are left as-is (we trust callers not to
    stuff a token into an int). The depth limit keeps a malicious
    self-referential dict from looping forever.
    """
    if data is None or depth > 6:
        return data
    if isinstance(data, Mapping):
        out: dict[str, Any] = {}
        for key, value in data.items():
            if (
                isinstance(key, str)
                and _SENSITIVE_KEY_RE.match(key)
                and isinstance(value, str)
            ):
                out[key] = redact_token(value)
            else:
                out[key] = redact_dict(value, depth + 1)
        return out
    if isinstance(data, (list, tuple)):
        items = [redact_dict(v, depth + 1) for v in data]
        return items if isinstance(data, list) else tuple(items)
    return data


# ---------------------------------------------------------------------------
# Structured kv emission
# ---------------------------------------------------------------------------


def kv(event: str, **fields: Any) -> str:
    """Render a structured log message as ``event key=value key=value``.

    Values pass through :func:`redact_dict` first, so accidentally
    logging ``token=hf_xyz`` will print ``token=hf_x…xyz`` instead of
    the full bearer.
    """
    if not fields:
        return event
    safe = redact_dict(fields)
    parts = [event]
    for key, value in safe.items():  # type: ignore[union-attr]
        parts.append(f"{key}={value}")
    return " ".join(parts)


def kv_log(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields: Any,
) -> None:
    """Emit a structured kv log line at the given level.

    Convenience wrapper around ``logger.log(level, kv(event, ...))`` that
    keeps call sites short and greppable.
    """
    logger.log(level, kv(event, **fields))
