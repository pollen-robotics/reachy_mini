from fastapi import Request, WebSocket

from ..backend.abstract import Backend
from ..daemon import Daemon


def get_daemon(request: Request) -> Daemon:
    return request.app.state.daemon


def get_backend(request: Request) -> Backend:
    return request.app.state.daemon.backend


def ws_get_backend(websocket: WebSocket) -> Backend:
    return websocket.app.state.daemon.backend
