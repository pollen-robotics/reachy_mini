"""FastAPI common request dependencies."""

from fastapi import HTTPException, Request, WebSocket

from ...apps.manager import AppManager
from ...nfc import NfcReader
from ..backend.abstract import Backend
from ..daemon import Daemon


def get_daemon(request: Request) -> Daemon:
    """Get the daemon as request dependency."""
    assert isinstance(request.app.state.daemon, Daemon)
    return request.app.state.daemon


def get_backend(request: Request) -> Backend:
    """Get the backend as request dependency."""
    backend = request.app.state.daemon.backend

    if backend is None or not backend.ready.is_set():
        raise HTTPException(status_code=503, detail="Backend not running")

    assert isinstance(backend, Backend)
    return backend


def get_app_manager(request: Request) -> "AppManager":
    """Get the app manager as request dependency."""
    assert isinstance(request.app.state.app_manager, AppManager)
    return request.app.state.app_manager


def get_nfc_reader(request: Request) -> NfcReader | None:
    """Get the NFC reader as request dependency (None if disabled)."""
    return getattr(request.app.state, "nfc_reader", None)


def ws_get_backend(websocket: WebSocket) -> Backend:
    """Get the backend as websocket dependency."""
    backend = websocket.app.state.daemon.backend

    if backend is None or not backend.ready.is_set():
        raise HTTPException(status_code=503, detail="Backend not running")

    assert isinstance(backend, Backend)
    return backend
