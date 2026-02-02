"""WebRTC signaling proxy router.

Provides a WebSocket endpoint that proxies signaling messages to the
internal GStreamer webrtcsink signaling server.
"""

import asyncio
import logging
from typing import Optional

import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webrtc", tags=["webrtc"])

# GStreamer webrtcsink's built-in signaling server
GSTREAMER_SIGNALING_URI = "ws://127.0.0.1:8443"


@router.websocket("/ws")
async def websocket_signaling_proxy(client_ws: WebSocket):
    """Proxy WebSocket signaling to GStreamer's internal signaling server.

    This endpoint allows external clients to connect to the WebRTC signaling
    via the daemon's HTTP server, which can be more easily proxied/secured.
    """
    await client_ws.accept()
    logger.info("Client connected to signaling proxy")

    gst_ws: Optional[websockets.WebSocketClientProtocol] = None

    try:
        # Connect to GStreamer's signaling server
        gst_ws = await websockets.connect(GSTREAMER_SIGNALING_URI)
        logger.info("Connected to GStreamer signaling server")

        async def forward_to_client():
            """Forward messages from GStreamer to client."""
            try:
                async for message in gst_ws:
                    logger.debug(f"GStreamer -> Client: {message[:100]}...")
                    await client_ws.send_text(message)
            except websockets.ConnectionClosed:
                logger.info("GStreamer connection closed")
            except Exception as e:
                logger.error(f"Error forwarding to client: {e}")

        async def forward_to_gstreamer():
            """Forward messages from client to GStreamer."""
            try:
                while True:
                    message = await client_ws.receive_text()
                    logger.debug(f"Client -> GStreamer: {message[:100]}...")
                    await gst_ws.send(message)
            except WebSocketDisconnect:
                logger.info("Client disconnected")
            except Exception as e:
                logger.error(f"Error forwarding to GStreamer: {e}")

        # Run both forwarding tasks concurrently
        await asyncio.gather(
            forward_to_client(),
            forward_to_gstreamer(),
            return_exceptions=True
        )

    except websockets.ConnectionRefused:
        logger.error("Could not connect to GStreamer signaling server - is WebRTC running?")
        await client_ws.close(code=1011, reason="WebRTC signaling server not available")
    except Exception as e:
        logger.error(f"Signaling proxy error: {e}")
        await client_ws.close(code=1011, reason=str(e))
    finally:
        if gst_ws:
            await gst_ws.close()
        logger.info("Signaling proxy connection closed")
