"""A small JSON-RPC 2.0 server an app mounts on its FastAPI to expose control.

An app registers method handlers and mounts this on a ``/rpc`` WebSocket. Every
connected peer — the daemon relay (for remote WebRTC/WS clients) *and* the app's
own local browser UI — speaks the same JSON-RPC there, so there is one control
protocol whether a client is on the LAN or remote.

Events are pushed as JSON-RPC notifications to every connected peer via
:meth:`JsonRpcServer.broadcast` (async) or :meth:`broadcast_threadsafe` (from a
realtime/audio thread).

Usage::

    rpc = JsonRpcServer()

    @rpc.method("conversation.status")
    async def _status(params):
        return {"phase": "running", ...}

    rpc.mount(settings_app)                 # adds WS route at /rpc
    rpc.broadcast_threadsafe("conversation.turn", {"state": "speaking"})
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Optional, TypeVar, Union, cast

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from reachy_mini.io.jsonrpc import (
    JsonRpcError,
    RpcNotification,
    error_from_exc,
    make_result,
    parse_request,
)

logger = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any]], Union[Any, Awaitable[Any]]]
# Unbounded so the decorator preserves each handler's exact signature (keeps
# --disallow-untyped-decorators happy); the handler is cast to Handler on
# registration.
F = TypeVar("F")


class JsonRpcServer:
    """Dispatches JSON-RPC methods and fans notifications out to all peers."""

    def __init__(self) -> None:
        """Create an empty server (register methods, then :meth:`mount`)."""
        self._methods: dict[str, Handler] = {}
        self._clients: set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # -- registration ------------------------------------------------------

    def method(self, name: str) -> Callable[[F], F]:
        """Register ``name`` -> handler(params) (decorator; sync or async)."""

        def _register(fn: F) -> F:
            self.register(name, cast(Handler, fn))
            return fn

        return _register

    def register(self, name: str, handler: Handler) -> None:
        """Register a method handler by name."""
        self._methods[name] = handler

    # -- notifications (events) -------------------------------------------

    async def broadcast(
        self, method: str, params: Optional[dict[str, Any]] = None
    ) -> None:
        """Push a notification to every connected peer (call on the app loop)."""
        # Serialize once, on the event hot path, in a single pydantic-core step
        # (no dict round-trip through the Python json module).
        payload = RpcNotification(method=method, params=params or {}).model_dump_json()
        for ws in list(self._clients):
            try:
                await ws.send_text(payload)
            except Exception:
                self._clients.discard(ws)

    def broadcast_threadsafe(
        self, method: str, params: Optional[dict[str, Any]] = None
    ) -> None:
        """Push a notification from any thread (e.g. the realtime handler)."""
        loop = self._loop
        if loop is None:
            return  # no peer connected yet; events while idle are dropped by design
        asyncio.run_coroutine_threadsafe(self.broadcast(method, params), loop)

    # -- mounting ----------------------------------------------------------

    def mount(self, app: FastAPI, path: str = "/rpc") -> None:
        """Add the ``/rpc`` WebSocket route to a FastAPI app."""

        @app.websocket(path)
        async def _rpc_ws(websocket: WebSocket) -> None:  # pragma: no cover - I/O
            await self._serve(websocket)

    async def _serve(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._loop = asyncio.get_running_loop()
        self._clients.add(websocket)
        try:
            while True:
                raw = await websocket.receive_text()
                await self._dispatch(websocket, raw)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning("[/rpc] connection error: %s", e)
        finally:
            self._clients.discard(websocket)

    async def _dispatch(self, websocket: WebSocket, raw: str) -> None:
        try:
            req = parse_request(raw)
        except JsonRpcError as e:
            await websocket.send_text(json.dumps(error_from_exc(None, e)))
            return

        handler = self._methods.get(req.method)
        if handler is None:
            if not req.is_notification:
                await websocket.send_text(
                    json.dumps(
                        error_from_exc(
                            req.id,
                            JsonRpcError(
                                f"unknown method: {req.method}",
                                reason="method_not_found",
                                code=-32601,
                            ),
                        )
                    )
                )
            return

        try:
            result = handler(req.params)
            if asyncio.iscoroutine(result):
                result = await result
        except Exception as e:  # noqa: BLE001 - report every failure to the caller
            if not req.is_notification:
                await websocket.send_text(json.dumps(error_from_exc(req.id, e)))
            else:
                logger.warning("[/rpc] notification %s failed: %s", req.method, e)
            return

        if not req.is_notification:
            await websocket.send_text(json.dumps(make_result(req.id, result)))
