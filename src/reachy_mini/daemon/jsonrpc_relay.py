"""Daemon-side JSON-RPC relay: the one control surface for apps over WebRTC/WS.

Incoming JSON-RPC frames from any client transport (WebRTC DataChannel,
``/ws/sdk``) are routed here by namespace:

* ``apps.*`` — handled locally against the :class:`AppManager` (start / stop /
  status of the single running app). This *is* the "apps API" exposed over
  WebRTC.
* anything else (e.g. ``conversation.*``) — relayed to the running app's
  ``/rpc`` WebSocket. The relay holds one persistent connection to that
  endpoint, which doubles as the daemon's subscription to the app's event
  stream: notifications the app pushes are re-broadcast to every connected
  client.

The relay runs on the daemon's main asyncio loop (where the AppManager and the
app WS client live); transports schedule :meth:`handle` onto it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from websockets.asyncio.client import ClientConnection, connect

from reachy_mini.apps.manager import AppManager
from reachy_mini.daemon.app.startup_app import ensure_startup_app_installed
from reachy_mini.io.jsonrpc import (
    JsonRpcError,
    RpcErrorResponse,
    RpcRequest,
    RpcSuccess,
    error_from_exc,
    make_error,
    make_result,
    parse_inbound,
    parse_request,
)

Reply = Callable[[dict[str, Any]], None]

_CONNECT_TIMEOUT_S = 5.0


class JsonRpcRelay:
    """Routes JSON-RPC control frames to the daemon or the running app."""

    def __init__(
        self,
        app_manager: AppManager,
        broadcast: Callable[[str], None],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Create the relay.

        Args:
            app_manager: source of app lifecycle + the running app URL.
            broadcast: sends a JSON string to *every* connected client (the
                backend's ``broadcast_to_all_clients``). Used to fan app
                notifications out.
            logger: optional logger.

        """
        self._apps = app_manager
        self._broadcast = broadcast
        self.logger = logger or logging.getLogger("reachy_mini.jsonrpc_relay")

        self._app_ws: Optional[ClientConnection] = None
        self._app_ws_lock = asyncio.Lock()
        self._reader_task: Optional[asyncio.Task[None]] = None
        # relay-assigned id -> caller reply callback, for correlating app replies,
        # plus the caller's original id to restore on the way back.
        self._pending: dict[str, Reply] = {}
        self._pending_original_id: dict[str, Any] = {}
        self._counter = 0

    # -- entry point -------------------------------------------------------

    async def handle(self, raw: str, reply: Reply) -> None:
        """Route one incoming frame; ``reply`` sends a response to its sender."""
        try:
            req = parse_request(raw)
        except JsonRpcError as e:
            reply(error_from_exc(None, e))
            return

        try:
            if req.namespace == "apps":
                await self._handle_apps(req, reply)
            else:
                await self._relay_to_app(req, reply)
        except JsonRpcError as e:
            if not req.is_notification:
                reply(error_from_exc(req.id, e))
        except Exception as e:  # noqa: BLE001 - surface every failure to the caller
            self.logger.warning("relay error on %s: %s", req.method, e)
            if not req.is_notification:
                reply(error_from_exc(req.id, e))

    async def aclose(self) -> None:
        """Close the app connection and fail any pending calls."""
        await self._drop_app_ws()

    # -- apps.* (daemon-local) --------------------------------------------

    async def _handle_apps(self, req: RpcRequest, reply: Reply) -> None:
        verb = req.method.split(".", 1)[1] if "." in req.method else ""
        if verb == "start":
            name = req.params.get("name")
            if not isinstance(name, str) or not name:
                raise JsonRpcError(
                    "apps.start requires a 'name'", reason="invalid_params", code=-32602
                )
            try:
                # keep_remote: the requester is this connected client that will
                # drive the app; don't evict its own WebRTC/DataChannel session.
                status = await self._apps.start_app(name, keep_remote=True)
            except RuntimeError as e:
                # start_app raises when an app already holds the robot slot.
                raise JsonRpcError(str(e), reason="already_running") from e
            result: Any = _status_dict(status)
        elif verb == "stop":
            await self._apps.stop_current_app()
            result = {"stopped": True}
        elif verb == "install":
            name = req.params.get("name")
            if not isinstance(name, str) or not name:
                raise JsonRpcError(
                    "apps.install requires a 'name'",
                    reason="invalid_params",
                    code=-32602,
                )
            # Install-if-missing from the HF catalog (no-op when already
            # installed). Can take minutes on a first install — callers
            # should use a generous timeout. Runs as its own task, so
            # other RPCs (apps.status polling, ...) keep answering.
            if not await ensure_startup_app_installed(self._apps, name):
                raise JsonRpcError(
                    f"could not install app {name!r} (not in the catalog, "
                    "or the install failed — see the daemon logs)",
                    reason="install_failed",
                )
            result = {"installed": True}
        elif verb == "status":
            result = self._current_status_dict()
        else:
            raise JsonRpcError(
                f"unknown method: {req.method}",
                reason="method_not_found",
                code=-32601,
            )
        if not req.is_notification:
            reply(make_result(req.id, result))

    def _current_status_dict(self) -> dict[str, Any]:
        app = self._apps.current_app
        if app is None or not self._apps.is_app_running():
            return {"state": "idle", "info": None, "error": None}
        return _status_dict(app.status)

    # -- relay to the running app's /rpc ----------------------------------

    async def _relay_to_app(self, req: RpcRequest, reply: Reply) -> None:
        ws = await self._ensure_app_ws()

        if req.is_notification:
            # Fire-and-forget: forward verbatim, nothing to correlate.
            await ws.send(req.model_dump_json())
            return

        self._counter += 1
        relay_id = f"relay-{self._counter}"
        # Record both correlation entries BEFORE the await: the reader task runs
        # on this same loop and can process the app's reply *during*
        # ``ws.send`` (a fast reply on the loopback WS). If `original_id` were
        # set after the await, that reply would restore id=None and the caller
        # would hang until timeout.
        self._pending[relay_id] = reply
        self._pending_original_id[relay_id] = req.id
        # Forward the request under a relay-assigned id (model_copy keeps every
        # other field intact); the reader restores `original_id` on the reply.
        outgoing = req.model_copy(update={"id": relay_id})
        try:
            await ws.send(outgoing.model_dump_json())
        except Exception as e:
            self._pending.pop(relay_id, None)
            self._pending_original_id.pop(relay_id, None)
            raise JsonRpcError(f"app unavailable: {e}", reason="app_unavailable") from e

    async def _ensure_app_ws(self) -> ClientConnection:
        async with self._app_ws_lock:
            if self._app_ws is not None:
                return self._app_ws

            url = self._apps.get_running_app_url()
            if not url:
                raise JsonRpcError(
                    "no app is running", reason="not_running", code=-32000
                )
            ws_url = _rpc_ws_url(url)
            try:
                self._app_ws = await asyncio.wait_for(
                    connect(ws_url), timeout=_CONNECT_TIMEOUT_S
                )
            except Exception as e:
                raise JsonRpcError(
                    f"cannot reach app /rpc at {ws_url}: {e}",
                    reason="app_unavailable",
                ) from e
            self._reader_task = asyncio.create_task(self._read_app(self._app_ws))
            self.logger.info("relay connected to app /rpc at %s", ws_url)
            return self._app_ws

    async def _read_app(self, ws: ClientConnection) -> None:
        try:
            async for message in ws:
                text = message if isinstance(message, str) else message.decode()
                self._on_app_frame(text)
        except Exception as e:
            self.logger.info("app /rpc reader ended: %s", e)
        finally:
            await self._drop_app_ws()

    def _on_app_frame(self, text: str) -> None:
        try:
            msg = parse_inbound(text)
        except JsonRpcError:
            return  # unparseable frame from the app — drop it
        # A correlated response (success or error) carries a relay-assigned id
        # (always a `relay-N` str) we're waiting on; restore the caller's id and
        # hand it back.
        if (
            isinstance(msg, (RpcSuccess, RpcErrorResponse))
            and isinstance(msg.id, str)
            and msg.id in self._pending
        ):
            reply = self._pending.pop(msg.id)
            original_id = self._pending_original_id.pop(msg.id, None)
            reply(msg.model_copy(update={"id": original_id}).model_dump())
        else:
            # A notification/event (or an uncorrelated response) — fan it out to
            # every client verbatim.
            self._broadcast(text)

    async def _drop_app_ws(self) -> None:
        ws, self._app_ws = self._app_ws, None
        task, self._reader_task = self._reader_task, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        # Fail every in-flight call so callers don't hang.
        pending, self._pending = self._pending, {}
        originals, self._pending_original_id = self._pending_original_id, {}
        for rid, reply in pending.items():
            reply(
                make_error(
                    originals.get(rid),
                    message="app connection lost",
                    reason="app_unavailable",
                )
            )


def _rpc_ws_url(app_url: str) -> str:
    """Turn an app's ``custom_app_url`` into its ``/rpc`` WebSocket URL.

    ``http://0.0.0.0:7860/`` -> ``ws://127.0.0.1:7860/rpc``. Binding on
    ``0.0.0.0`` means "listen on all interfaces"; the daemon reaches it on the
    loopback.
    """
    p = urlparse(app_url)
    host = "127.0.0.1" if p.hostname in (None, "0.0.0.0") else p.hostname
    scheme = "wss" if p.scheme == "https" else "ws"
    port = f":{p.port}" if p.port else ""
    return f"{scheme}://{host}{port}/rpc"


def _status_dict(status: Any) -> dict[str, Any]:
    """Serialize an ``AppStatus`` to a plain JSON dict."""
    info = status.info
    return {
        "state": str(
            status.state.value if hasattr(status.state, "value") else status.state
        ),
        "error": status.error,
        "info": {"name": info.name, "description": info.description, "url": info.url},
    }
