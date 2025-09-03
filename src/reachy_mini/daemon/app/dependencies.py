"""FastAPI common request dependencies."""

from fastapi import Request, WebSocket

from ..backend.abstract import Backend
from ..daemon import Daemon


def get_daemon(request: Request) -> Daemon:
    """Get the daemon as request dependency."""
    return request.app.state.daemon


def get_backend(request: Request) -> Backend:
    """Get the backend as request dependency."""
    return request.app.state.daemon.backend


def ws_get_backend(websocket: WebSocket) -> Backend:
    """Get the backend as websocket dependency."""
    return websocket.app.state.daemon.backend
