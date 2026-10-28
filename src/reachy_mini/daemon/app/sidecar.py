"""The Python half of the Rust daemon (reachy_mini_daemon), on loopback.

The Rust daemon owns the robot, its media and its clients. What it doesn't
do yet runs here, in a process it spawns: the apps, HF sign-in, telemetry
settings, update, Wi-Fi and cache routes, which it proxies, and the JSON-RPC
frames its clients send (``apps.*``, a running app's own methods), which it
forwards over ``/internal/rpc``. The managed app slot stays the Rust
daemon's: :class:`RemoteRobotAppLock` asks it over loopback.

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
from typing import Any, AsyncGenerator, Optional

import uvicorn
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect

from reachy_mini.apps.manager import AppManager
from reachy_mini.daemon.app.routers import apps, hf_auth, telemetry
from reachy_mini.daemon.jsonrpc_relay import JsonRpcRelay

logger = logging.getLogger(__name__)

_LOCK_TIMEOUT_S = 2.0


class RemoteRobotAppLock:
    """RobotAppLock's local-app side, held by the Rust daemon.

    Only what :class:`AppManager` calls. The remote side (sessions through
    central) and the idle reset on release live in the Rust daemon.
    """

    def __init__(self, daemon_url: str) -> None:
        """Ask the daemon at ``daemon_url`` (its ``/internal/app-lock``)."""
        self._url = f"{daemon_url.rstrip('/')}/internal/app-lock"

    def _post(self, verb: str, body: dict[str, Any]) -> bool:
        """POST to the lock; False when the slot is taken (409)."""
        request = urllib.request.Request(
            f"{self._url}/{verb}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=_LOCK_TIMEOUT_S):
                return True
        except urllib.error.HTTPError as e:
            if e.code == 409:
                return False
            raise

    def try_acquire_local(self, app_name: str) -> bool:
        """Take the slot only if free (no local app, no remote session)."""
        return self._post("acquire", {"name": app_name, "mode": "try"})

    async def acquire_local_evicting_remote(self, app_name: str) -> None:
        """Take the slot; the daemon has a remote session holding it ended."""
        if not await asyncio.to_thread(
            self._post, "acquire", {"name": app_name, "mode": "evict"}
        ):
            raise RuntimeError("A local app is already running")

    def acquire_local_keeping_remote(self, app_name: str) -> None:
        """Take the slot for an app the connected remote client drives."""
        if not self._post("acquire", {"name": app_name, "mode": "keep"}):
            raise RuntimeError("A local app is already running")

    def release_local(self, app_name: Optional[str] = None) -> None:
        """Give the slot back. Never raises: it runs as an app exits."""
        try:
            self._post("release", {"name": app_name})
        except Exception as e:  # noqa: BLE001 - the app is gone either way
            logger.warning("app lock release failed: %s", e)


@dataclass
class SidecarDaemon:
    """What AppManager and the routers read off the Python daemon."""

    wireless_version: bool
    desktop_app_daemon: bool
    robot_app_lock: RemoteRobotAppLock
    backend: None = None

    async def start_central_relay_if_running(self) -> None:
        """No relay here: the signalling server reads the new token itself."""


def create_app(
    wireless_version: bool, desktop_app_daemon: bool, daemon_url: str
) -> FastAPI:
    """Build the sidecar: its routes and the JSON-RPC endpoint the daemon dials."""
    daemon = SidecarDaemon(
        wireless_version=wireless_version,
        desktop_app_daemon=desktop_app_daemon,
        robot_app_lock=RemoteRobotAppLock(daemon_url),
    )
    app_manager = AppManager(
        wireless_version=wireless_version,
        desktop_app_daemon=desktop_app_daemon,
        daemon=daemon,  # type: ignore[arg-type]
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
        """JSON-RPC from the daemon's clients.

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
