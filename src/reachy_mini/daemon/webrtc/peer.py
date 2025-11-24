"""WebRTC peer connection handler using aiortc."""

import asyncio
import json
import logging
import uuid
from typing import Any, Callable, Coroutine

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel
from aiortc.contrib.media import MediaPlayer, MediaRelay

logger = logging.getLogger(__name__)


class WebRTCPeer:
    """Manages a single WebRTC peer connection with data channels."""

    def __init__(
        self,
        peer_id: str,
        on_message: Callable[[str, str], Coroutine[Any, Any, str]],
        on_close: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize WebRTC peer.

        Args:
            peer_id: Unique identifier for this peer connection
            on_message: Async callback for handling incoming messages.
                       Receives (peer_id, message) and returns response string.
            on_close: Optional callback when connection closes
        """
        self._peer_id = peer_id
        self._on_message = on_message
        self._on_close = on_close
        self._pc = RTCPeerConnection()
        self._data_channel: RTCDataChannel | None = None
        self._closed = False

        self._setup_handlers()

    @property
    def peer_id(self) -> str:
        """Get the peer ID."""
        return self._peer_id

    @property
    def is_connected(self) -> bool:
        """Check if peer is connected."""
        return (
            not self._closed
            and self._pc.connectionState in ("connected", "connecting")
        )

    def _setup_handlers(self) -> None:
        """Set up event handlers for the peer connection."""

        @self._pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            logger.info(f"Peer {self._peer_id} connection state: {self._pc.connectionState}")
            if self._pc.connectionState == "failed":
                await self.close()
            elif self._pc.connectionState == "closed":
                if self._on_close and not self._closed:
                    self._on_close(self._peer_id)
                self._closed = True

        @self._pc.on("datachannel")
        def on_datachannel(channel: RTCDataChannel) -> None:
            logger.info(f"Peer {self._peer_id} data channel received: {channel.label}")
            self._data_channel = channel
            self._setup_data_channel(channel)

    def _setup_data_channel(self, channel: RTCDataChannel) -> None:
        """Set up handlers for a data channel."""

        @channel.on("open")
        def on_open() -> None:
            logger.info(f"Peer {self._peer_id} data channel '{channel.label}' opened")

        @channel.on("close")
        def on_close() -> None:
            logger.info(f"Peer {self._peer_id} data channel '{channel.label}' closed")

        @channel.on("message")
        async def on_message(message: str) -> None:
            logger.debug(f"Peer {self._peer_id} received: {message[:100]}...")
            try:
                response = await self._on_message(self._peer_id, message)
                if response and channel.readyState == "open":
                    channel.send(response)
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                error_response = json.dumps({
                    "error": str(e),
                    "type": "error"
                })
                if channel.readyState == "open":
                    channel.send(error_response)

    async def create_offer(self) -> dict[str, str]:
        """Create an SDP offer (for manual pairing flow).

        In manual pairing, the server creates the offer and the client creates the answer.
        This is the reverse of the typical WebRTC flow.

        Returns:
            Dictionary with "sdp" and "type" keys for the offer
        """
        # Create a data channel (required before creating offer)
        self._data_channel = self._pc.createDataChannel("reachy-control")
        self._setup_data_channel(self._data_channel)

        # Create offer
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)

        return {
            "sdp": self._pc.localDescription.sdp,
            "type": self._pc.localDescription.type
        }

    async def handle_offer(self, sdp: str, sdp_type: str = "offer") -> dict[str, str]:
        """Handle an incoming SDP offer and return an answer.

        Args:
            sdp: The SDP offer string
            sdp_type: The SDP type (usually "offer")

        Returns:
            Dictionary with "sdp" and "type" keys for the answer
        """
        offer = RTCSessionDescription(sdp=sdp, type=sdp_type)
        await self._pc.setRemoteDescription(offer)

        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)

        return {
            "sdp": self._pc.localDescription.sdp,
            "type": self._pc.localDescription.type
        }

    async def handle_answer(self, sdp: str, sdp_type: str = "answer") -> None:
        """Handle an incoming SDP answer (for manual pairing flow).

        Args:
            sdp: The SDP answer string
            sdp_type: The SDP type (usually "answer")
        """
        answer = RTCSessionDescription(sdp=sdp, type=sdp_type)
        await self._pc.setRemoteDescription(answer)

    async def add_ice_candidate(self, candidate: dict[str, Any]) -> None:
        """Add an ICE candidate from the remote peer.

        Args:
            candidate: ICE candidate dictionary with candidate, sdpMid, sdpMLineIndex
        """
        from aiortc import RTCIceCandidate

        if candidate.get("candidate"):
            ice_candidate = RTCIceCandidate(
                candidate["candidate"],
                candidate.get("sdpMid", ""),
                candidate.get("sdpMLineIndex", 0)
            )
            await self._pc.addIceCandidate(ice_candidate)

    def send(self, message: str) -> bool:
        """Send a message through the data channel.

        Args:
            message: The message to send

        Returns:
            True if sent successfully, False otherwise
        """
        if self._data_channel and self._data_channel.readyState == "open":
            self._data_channel.send(message)
            return True
        return False

    async def close(self) -> None:
        """Close the peer connection."""
        if not self._closed:
            self._closed = True
            await self._pc.close()
            if self._on_close:
                self._on_close(self._peer_id)
            logger.info(f"Peer {self._peer_id} closed")
