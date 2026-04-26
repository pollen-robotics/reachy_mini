"""Starlette middleware that propagates ``X-Trace-Id`` into log context.

When the mobile app issues a daemon request (over LAN HTTP or via the
WebRTC ``http_proxy`` tunnel), it tags each request with
``X-Trace-Id``. This middleware:

1. Reads that header (or mints a new id when absent, e.g. for
   browser-originated requests like the dashboard or for ad-hoc curl
   testing).
2. Sets it on the :data:`logging_ctx.trace_id_var` ContextVar.
3. Logs a one-liner per request with the request method/path/status
   and elapsed time, so we have a stable observation point even on
   routers that haven't been instrumented yet.
4. Echoes the trace back as ``X-Trace-Id`` on the response, so the
   mobile app can confirm correlation in cases where it didn't set
   one (e.g. dashboard Open in Browser).

We deliberately filter out a small set of high-frequency polling
paths from the per-request log (relay-status, health-check) so DEBUG
output isn't dominated by them.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from .logging_ctx import (
    kv,
    new_trace_id,
    reset_trace_id,
    set_trace_id,
)

_logger = logging.getLogger(__name__)

# Keep the per-request log free of the highest-frequency endpoints.
# They already have their own structured events when something
# interesting happens; otherwise they're pure noise at INFO level.
_QUIET_PATHS = {
    "/health-check",
    "/api/hf-auth/relay-status",
    "/api/hf-auth/central-robot-status",
    "/api/daemon/status",
    "/api/state",
}


class TraceIdMiddleware(BaseHTTPMiddleware):
    """ASGI middleware: scope ``trace_id`` per request.

    Subclasses :class:`BaseHTTPMiddleware` so we can ``await
    call_next`` and observe the response. The overhead of this base
    class (~50µs per request) is negligible against the network
    latencies we care about, and the API is simpler than implementing
    raw ASGI.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Bind to the wrapped ASGI app."""
        super().__init__(app)

    async def dispatch(  # type: ignore[override]
        self, request: Request, call_next: Any
    ) -> Response:
        """Scope ``trace_id`` to this request and emit a one-line summary."""
        trace = request.headers.get("X-Trace-Id") or new_trace_id()
        token = set_trace_id(trace)
        t0 = time.perf_counter()
        path = request.url.path
        try:
            response: Response = await call_next(request)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            if path not in _QUIET_PATHS:
                level = logging.WARNING if response.status_code >= 400 else logging.INFO
                _logger.log(
                    level,
                    kv(
                        "http.request",
                        method=request.method,
                        path=path,
                        status=response.status_code,
                        latency_ms=elapsed_ms,
                    ),
                )
            response.headers["X-Trace-Id"] = trace
            return response
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            _logger.exception(
                kv(
                    "http.error",
                    method=request.method,
                    path=path,
                    latency_ms=elapsed_ms,
                    exc_type=type(exc).__name__,
                )
            )
            raise
        finally:
            reset_trace_id(token)
