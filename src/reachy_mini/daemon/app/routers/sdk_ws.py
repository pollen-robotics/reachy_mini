"""WebSocket endpoint for SDK communication."""

from fastapi import APIRouter, WebSocket, WebSocketException

router = APIRouter()


@router.websocket("/ws/sdk")
async def ws_sdk(websocket: WebSocket) -> None:
    """Handle SDK WebSocket connections."""
    ws_server = websocket.app.state.daemon.ws_server
    if ws_server is None:
        raise WebSocketException(code=1013, reason="Daemon not ready")
    await ws_server.handle_client(websocket)
