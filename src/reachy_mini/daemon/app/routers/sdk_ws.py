"""WebSocket endpoint for SDK communication."""

from fastapi import APIRouter, WebSocket

router = APIRouter()


@router.websocket("/ws/sdk")
async def ws_sdk(websocket: WebSocket) -> None:
    """Handle SDK WebSocket connections."""
    ws_server = websocket.app.state.daemon.ws_server
    await ws_server.handle_client(websocket)
