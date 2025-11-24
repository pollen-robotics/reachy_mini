"""WebRTC signaling router for Reachy Mini daemon.

Provides HTTP endpoints for WebRTC signaling to establish
DataChannel connections from HTTPS frontends.
"""

import uuid
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from reachy_mini.daemon.webrtc import ConnectionManager, WebRTCPeer
from reachy_mini.daemon.webrtc.manager import MessageHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webrtc", tags=["webrtc"])


class SDPOffer(BaseModel):
    """SDP offer from client."""

    sdp: str | None = None
    type: str = "offer"
    peer_id: str | None = None


class SDPAnswer(BaseModel):
    """SDP answer to client."""

    sdp: str
    type: str
    peer_id: str


class ICECandidate(BaseModel):
    """ICE candidate exchange."""

    candidate: str
    sdpMid: str | None = None
    sdpMLineIndex: int | None = None
    peer_id: str


class ConnectionInfo(BaseModel):
    """Connection status information."""

    peer_id: str
    connected: bool


# Module-level connection manager (initialized on first use)
_connection_manager: ConnectionManager | None = None


def get_connection_manager(request: Request) -> ConnectionManager:
    """Get or create the connection manager.

    Args:
        request: FastAPI request object

    Returns:
        The ConnectionManager instance
    """
    global _connection_manager

    if _connection_manager is None:
        _connection_manager = ConnectionManager()

        # Set up message handler with backend access
        def get_backend() -> Any:
            daemon = request.app.state.daemon
            if daemon.backend is None:
                raise RuntimeError("Daemon backend not initialized")
            return daemon.backend

        handler = MessageHandler(get_backend)
        _connection_manager.set_message_handler(handler)

    return _connection_manager


@router.post("/offer")
async def handle_offer(offer: SDPOffer, request: Request) -> dict[str, Any]:
    """Handle WebRTC offer or create one.

    Two modes:
    1. If sdp is provided: traditional flow - client sends offer, server returns answer
    2. If sdp is None: manual pairing flow - server creates offer for client

    Args:
        offer: The SDP offer from the client (or empty for manual pairing)
        request: FastAPI request object

    Returns:
        SDP answer (mode 1) or SDP offer (mode 2)
    """
    manager = get_connection_manager(request)

    # Generate peer ID if not provided
    peer_id = offer.peer_id or str(uuid.uuid4())

    logger.info(f"Processing request from peer {peer_id}, mode={'offer' if offer.sdp else 'manual pairing'}")

    try:
        peer = await manager.create_peer(peer_id)

        if offer.sdp:
            # Traditional flow: handle client offer, return server answer
            answer = await peer.handle_offer(offer.sdp, offer.type)
            return {
                "sdp": answer["sdp"],
                "type": answer["type"],
                "peer_id": peer_id
            }
        else:
            # Manual pairing flow: create server offer for client
            offer_data = await peer.create_offer()
            return {
                "sdp": offer_data["sdp"],
                "type": offer_data["type"],
                "peer_id": peer_id
            }

    except Exception as e:
        logger.error(f"Error handling offer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/answer")
async def handle_answer(answer: SDPAnswer, request: Request) -> dict[str, str]:
    """Handle WebRTC answer from client (manual pairing flow).

    Args:
        answer: The SDP answer from the client
        request: FastAPI request object

    Returns:
        Status message
    """
    manager = get_connection_manager(request)
    peer = manager.get_peer(answer.peer_id)

    if not peer:
        raise HTTPException(status_code=404, detail=f"Peer {answer.peer_id} not found")

    logger.info(f"Received answer from peer {answer.peer_id}")

    try:
        await peer.handle_answer(answer.sdp, answer.type)
        return {"status": "ok", "peer_id": answer.peer_id}

    except Exception as e:
        logger.error(f"Error handling answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ice-candidate")
async def add_ice_candidate(candidate: ICECandidate, request: Request) -> dict[str, str]:
    """Add ICE candidate for a peer connection.

    Args:
        candidate: The ICE candidate from the client
        request: FastAPI request object

    Returns:
        Status message
    """
    manager = get_connection_manager(request)
    peer = manager.get_peer(candidate.peer_id)

    if not peer:
        raise HTTPException(status_code=404, detail=f"Peer {candidate.peer_id} not found")

    try:
        await peer.add_ice_candidate({
            "candidate": candidate.candidate,
            "sdpMid": candidate.sdpMid,
            "sdpMLineIndex": candidate.sdpMLineIndex
        })
        return {"status": "ok"}

    except Exception as e:
        logger.error(f"Error adding ICE candidate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_status(request: Request) -> dict[str, Any]:
    """Get WebRTC connection status.

    Args:
        request: FastAPI request object

    Returns:
        Connection status information
    """
    manager = get_connection_manager(request)

    return {
        "peer_count": manager.peer_count,
        "peer_ids": manager.peer_ids
    }


@router.get("/status/{peer_id}")
async def get_peer_status(peer_id: str, request: Request) -> ConnectionInfo:
    """Get status of a specific peer connection.

    Args:
        peer_id: The peer ID to check
        request: FastAPI request object

    Returns:
        Connection info for the peer
    """
    manager = get_connection_manager(request)
    peer = manager.get_peer(peer_id)

    if not peer:
        raise HTTPException(status_code=404, detail=f"Peer {peer_id} not found")

    return ConnectionInfo(
        peer_id=peer_id,
        connected=peer.is_connected
    )


@router.delete("/peer/{peer_id}")
async def close_peer(peer_id: str, request: Request) -> dict[str, str]:
    """Close a peer connection.

    Args:
        peer_id: The peer ID to close
        request: FastAPI request object

    Returns:
        Status message
    """
    manager = get_connection_manager(request)
    peer = manager.get_peer(peer_id)

    if not peer:
        raise HTTPException(status_code=404, detail=f"Peer {peer_id} not found")

    await manager.remove_peer(peer_id)
    return {"status": "closed", "peer_id": peer_id}
