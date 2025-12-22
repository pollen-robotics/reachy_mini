"""FastAPI common request dependencies."""

import os

from fastapi import HTTPException, Request, WebSocket
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ...apps.manager import AppManager
from ..backend.abstract import Backend
from ..daemon import Daemon

security = HTTPBearer(auto_error=False)


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


def ws_get_backend(websocket: WebSocket) -> Backend:
    """Get the backend as websocket dependency."""
    backend = websocket.app.state.daemon.backend

    if backend is None or not backend.ready.is_set():
        raise HTTPException(status_code=503, detail="Backend not running")

    assert isinstance(backend, Backend)
    return backend


def verify_api_token(credentials: HTTPAuthorizationCredentials = security) -> str:
    """Verify API token for sensitive endpoints.
    
    This checks for a valid Bearer token in the Authorization header.
    The token is validated against the REACHY_API_TOKEN environment variable.
    
    Args:
        credentials: HTTPAuthorizationCredentials from the request header.
        
    Returns:
        The validated token string.
        
    Raises:
        HTTPException: If no valid token is provided.
    """
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Missing authorization credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    expected_token = os.environ.get("REACHY_API_TOKEN")
    
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=403,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return credentials.credentials
