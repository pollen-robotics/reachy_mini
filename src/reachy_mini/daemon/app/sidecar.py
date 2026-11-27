"""The Python half of the Rust daemon (reachy_mini_daemon), on loopback.

The Rust daemon owns the robot, its media, its clients and the running app.
What it doesn't do yet runs here, in a process it spawns: installing,
listing and updating apps, HF sign-in, telemetry settings, update, Wi-Fi and
cache routes, which it proxies, and the ``apps.*`` JSON-RPC calls it doesn't
answer itself, which it forwards over ``/internal/rpc``. To change the files
of an app it may be running, :class:`DaemonApps` asks the daemon.

Run as ``python -m reachy_mini.daemon.app.sidecar``.
"""

import argparse
import asyncio
import json
import logging
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Optional

import uvicorn
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect

from reachy_mini.apps.manager import AppManager, AppStatus
from reachy_mini.daemon.app.routers import apps, hf_auth, telemetry
from reachy_mini.daemon.jsonrpc_relay import JsonRpcRelay

logger = logging.getLogger(__name__)

_STATUS_TIMEOUT_S = 2.0
# The daemon waits up to 20 s for the app, then returns the robot to zero.
_STOP_TIMEOUT_S = 30.0


class DaemonApps(AppManager):
    """AppManager whose running app is the Rust daemon's.

    Installs, removals and updates run here and must not touch a running
    app's files: the running app and stopping it come from the daemon's
    ``/api/apps`` routes. Starting apps is the daemon's alone.
    """

    def __init__(self, daemon_url: str, **kwargs: Any) -> None:
        """Ask the daemon at ``daemon_url`` about its app."""
        self._url = f"{daemon_url.rstrip('/')}/api/apps"
        super().__init__(**kwargs)

    @property
    def current_app(self) -> Any:
        """The daemon's app (only its ``status``), None if none or unreachable."""
        # shortcut: a blocking loopback GET on each read; fine at install time.
        try:
            with urllib.request.urlopen(
                f"{self._url}/current-app-status", timeout=_STATUS_TIMEOUT_S
            ) as r:
                status = json.loads(r.read())
        except (OSError, ValueError) as e:
            logger.warning("daemon app status unavailable: %s", e)
            return None
        if status is None:
            return None
        return SimpleNamespace(status=AppStatus.model_validate(status))

    @current_app.setter
    def current_app(self, value: Any) -> None:
        """Ignored: AppManager.__init__ sets it, the daemon owns it."""

    def _stop(self) -> None:
        request = urllib.request.Request(
            f"{self._url}/stop-current-app", data=b"", method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=_STOP_TIMEOUT_S):
                pass
        except urllib.error.HTTPError as e:
            raise RuntimeError(json.loads(e.read()).get("detail", str(e))) from e

    async def stop_current_app(self, timeout: float | None = 20.0) -> None:
        """Have the daemon stop its app; RuntimeError if none runs."""
        await asyncio.to_thread(self._stop)

    async def start_app(self, *args: Any, **kwargs: Any) -> AppStatus:
        """Apps start in the daemon (its /api/apps/start-app, apps.start)."""
        raise RuntimeError("Apps are started by the daemon")

    async def restart_current_app(self, keep_remote: bool = False) -> AppStatus:
        """Apps restart in the daemon."""
        raise RuntimeError("Apps are started by the daemon")

    async def close(self) -> None:
        """Do nothing: the daemon stops its app itself."""


@dataclass
class SidecarDaemon:
    """What AppManager and the routers read off the Python daemon."""

    wireless_version: bool
    desktop_app_daemon: bool
    backend: None = None

    async def start_central_relay_if_running(self) -> None:
        """No relay here: the signalling server reads the new token itself."""


def create_app(
    wireless_version: bool, desktop_app_daemon: bool, daemon_url: str
) -> FastAPI:
    """Build the sidecar: its routes and the JSON-RPC endpoint the daemon dials."""
    daemon = SidecarDaemon(
        wireless_version=wireless_version, desktop_app_daemon=desktop_app_daemon
    )
    app_manager = DaemonApps(
        daemon_url,
        wireless_version=wireless_version,
        desktop_app_daemon=desktop_app_daemon,
        daemon=daemon,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        yield
        await app_manager.close()

    app = FastAPI(lifespan=lifespan)
    app.state.daemon = daemon
    app.state.app_manager = app_manager
    app.state.startup_app_antenna_watcher_task = None

    router = APIRouter(prefix="/api")
    router.include_router(apps.router)
    router.include_router(hf_auth.router)
    router.include_router(telemetry.router)
    app.include_router(router)
    if wireless_version:
        from reachy_mini.daemon.app.routers import cache, update, wifi_config

        app.include_router(cache.router)
        app.include_router(update.router)
        app.include_router(wifi_config.router)

    @app.websocket("/internal/rpc")
    async def rpc(ws: WebSocket) -> None:
        """Answer the ``apps.*`` JSON-RPC the daemon doesn't.

        In: ``{"client", "frame"}``, a client's frame. Out: ``{"client",
        "frame"}``, the reply for that client, or ``client: null`` for a
        notification to every client.
        """
        await ws.accept()
        out: asyncio.Queue[str] = asyncio.Queue()

        def send(client: Optional[str], frame: str) -> None:
            out.put_nowait(json.dumps({"client": client, "frame": frame}))

        relay = JsonRpcRelay(app_manager, lambda frame: send(None, frame))
        calls: set[asyncio.Task[None]] = set()

        async def writer() -> None:
            while True:
                await ws.send_text(await out.get())

        writing = asyncio.create_task(writer())
        try:
            async for text in ws.iter_text():
                msg = json.loads(text)

                def reply(r: dict[str, Any], client: str = msg["client"]) -> None:
                    send(client, json.dumps(r))

                call = asyncio.create_task(relay.handle(msg["frame"], reply))
                calls.add(call)
                call.add_done_callback(calls.discard)
        except WebSocketDisconnect:
            pass
        finally:
            writing.cancel()
            await relay.aclose()

    return app


def main() -> None:
    """Serve the sidecar on loopback."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wireless-version", action="store_true")
    parser.add_argument("--desktop-app-daemon", action="store_true")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--daemon-url", default="http://127.0.0.1:8000")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    app = create_app(args.wireless_version, args.desktop_app_daemon, args.daemon_url)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
