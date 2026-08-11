"""Custom ASGI middleware for the daemon HTTP app."""

import json
import logging
from collections.abc import Iterable

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
