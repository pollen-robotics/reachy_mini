"""Central signaling relay for WebRTC.

Connects to a central signaling server and relays messages to/from
the local GStreamer webrtcsink signaling server.
"""

import asyncio
import json
import logging
from typing import Optional

import websockets
from websockets.asyncio.client import ClientConnection

logger = logging.getLogger(__name__)

# Central signaling server URL
CENTRAL_SIGNALING_SERVER = "wss://cduss-reachy-mini-central.hf.space/ws"
LOCAL_GSTREAMER_SIGNALING = "ws://127.0.0.1:8443"

# Reconnection settings
RECONNECT_INTERVAL = 5.0  # seconds


class CentralSignalingRelay:
    """Relay signaling messages between central server and local GStreamer."""

    def __init__(
        self,
        central_uri: str = CENTRAL_SIGNALING_SERVER,
        local_uri: str = LOCAL_GSTREAMER_SIGNALING,
        hf_token: Optional[str] = None,
        robot_name: str = "reachymini",
    ):
        """Initialize the relay.

        Args:
            central_uri: WebSocket URI of central signaling server
            local_uri: WebSocket URI of local GStreamer signaling server
            hf_token: HuggingFace token for authentication
            robot_name: Name to register as producer

        """
        self.central_uri = central_uri
        self.local_uri = local_uri
        self.hf_token = hf_token
        self.robot_name = robot_name

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._central_ws: Optional[ClientConnection] = None
        self._local_ws: Optional[ClientConnection] = None
        self._central_peer_id: Optional[str] = None
        self._local_peer_id: Optional[str] = None

        # Map central session IDs to local peer IDs
        self._session_to_local_peer: dict[str, str] = {}

    async def start(self) -> None:
        """Start the relay service."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Central signaling relay started")

    async def stop(self) -> None:
        """Stop the relay service."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        await self._close_connections()
        logger.info("Central signaling relay stopped")

    async def _close_connections(self) -> None:
        """Close all WebSocket connections."""
        if self._central_ws:
            try:
                await self._central_ws.close()
            except Exception:
                pass
            self._central_ws = None

        if self._local_ws:
            try:
                await self._local_ws.close()
            except Exception:
                pass
            self._local_ws = None

    async def _run_loop(self) -> None:
        """Main loop that maintains connections and relays messages."""
        while self._running:
            try:
                await self._connect_and_relay()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Relay error: {e}")

            if self._running:
                logger.info(f"Reconnecting in {RECONNECT_INTERVAL}s...")
                await asyncio.sleep(RECONNECT_INTERVAL)

    async def _connect_and_relay(self) -> None:
        """Connect to both servers and relay messages."""
        # First, check if we have a token
        if not self.hf_token:
            logger.debug("No HF token available, skipping central signaling")
            await asyncio.sleep(RECONNECT_INTERVAL)
            return

        # Connect to central server
        try:
            logger.info(f"Connecting to central server: {self.central_uri}")
            self._central_ws = await websockets.connect(
                self.central_uri,
                additional_headers={"Authorization": f"Bearer {self.hf_token}"},
                ping_interval=20,
                ping_timeout=10,
            )
            logger.info("Connected to central signaling server")
        except Exception as e:
            logger.warning(f"Failed to connect to central server: {e}")
            raise

        # Connect to local GStreamer signaling
        try:
            logger.debug(f"Connecting to local GStreamer: {self.local_uri}")
            self._local_ws = await websockets.connect(
                self.local_uri,
                ping_interval=20,
                ping_timeout=10,
            )
            logger.info("Connected to local GStreamer signaling")
        except Exception as e:
            logger.warning(f"Failed to connect to local GStreamer signaling: {e}")
            await self._central_ws.close()
            raise

        try:
            # Start message relay tasks
            await asyncio.gather(
                self._handle_central_messages(),
                self._handle_local_messages(),
            )
        finally:
            await self._close_connections()

    async def _handle_central_messages(self) -> None:
        """Handle messages from central server."""
        if not self._central_ws:
            return

        try:
            async for message in self._central_ws:
                try:
                    msg = json.loads(message)
                    await self._process_central_message(msg)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from central: {message[:100]}")
        except websockets.ConnectionClosed:
            logger.info("Central server connection closed")
        except Exception as e:
            logger.error(f"Error handling central messages: {e}")

    async def _handle_local_messages(self) -> None:
        """Handle messages from local GStreamer signaling."""
        if not self._local_ws:
            return

        try:
            async for message in self._local_ws:
                try:
                    msg = json.loads(message)
                    await self._process_local_message(msg)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from local: {message[:100]}")
        except websockets.ConnectionClosed:
            logger.info("Local GStreamer connection closed")
        except Exception as e:
            logger.error(f"Error handling local messages: {e}")

    async def _process_central_message(self, msg: dict) -> None:
        """Process a message from the central server."""
        msg_type = msg.get("type", "")
        logger.debug(f"Central -> Relay: {msg_type}")

        if msg_type == "welcome":
            # Received our peer ID from central server
            self._central_peer_id = msg.get("peerId")
            logger.info(f"Registered on central as: {self._central_peer_id}")

            # Register as producer
            await self._send_to_central({
                "type": "setPeerStatus",
                "roles": ["producer"],
                "meta": {"name": self.robot_name},
            })

        elif msg_type == "startSession":
            # A client wants to connect - forward to local GStreamer
            client_peer_id = msg.get("peerId")
            session_id = msg.get("sessionId")
            logger.info(f"Session request from client: {client_peer_id}")

            # We need to start a session on local GStreamer
            # The local GStreamer will send us an offer which we relay back
            # For now, store the session mapping
            if session_id and self._local_peer_id:
                self._session_to_local_peer[session_id] = client_peer_id

            # Request session with local GStreamer producer
            # First we need to know the local producer ID - request list
            await self._send_to_local({"type": "list"})

        elif msg_type == "peer":
            # SDP/ICE from client - relay to local GStreamer
            session_id = msg.get("sessionId")
            if session_id and self._local_ws:
                # Forward to local GStreamer
                # Transform message for local format if needed
                local_msg = {
                    "type": "peer",
                    "sessionId": session_id,
                }
                if "sdp" in msg:
                    local_msg["sdp"] = msg["sdp"]
                if "ice" in msg:
                    local_msg["ice"] = msg["ice"]

                await self._send_to_local(local_msg)

        elif msg_type == "endSession":
            session_id = msg.get("sessionId")
            if session_id:
                self._session_to_local_peer.pop(session_id, None)
                # Forward to local
                await self._send_to_local(msg)

    async def _process_local_message(self, msg: dict) -> None:
        """Process a message from local GStreamer signaling."""
        msg_type = msg.get("type", "")
        logger.debug(f"Local -> Relay: {msg_type}")

        if msg_type == "welcome":
            # Received our peer ID from local GStreamer
            self._local_peer_id = msg.get("peerId")
            logger.info(f"Connected to local GStreamer as: {self._local_peer_id}")

            # Register as listener to receive producer announcements
            await self._send_to_local({
                "type": "setPeerStatus",
                "roles": ["listener"],
                "meta": {"name": "central-relay"},
            })

        elif msg_type == "list":
            # List of local producers
            producers = msg.get("producers", [])
            if producers:
                # Store the first producer ID (should be the webrtcsink)
                self._local_producer_id = producers[0].get("id")
                logger.debug(f"Local producer found: {self._local_producer_id}")

        elif msg_type == "peerStatusChanged":
            # A local producer appeared/disappeared
            peer_id = msg.get("peerId")
            roles = msg.get("roles", [])
            if "producer" in roles:
                self._local_producer_id = peer_id
                logger.debug(f"Local producer registered: {peer_id}")

        elif msg_type == "sessionStarted":
            # Local session started
            session_id = msg.get("sessionId")
            logger.debug(f"Local session started: {session_id}")

        elif msg_type == "peer":
            # SDP/ICE from local GStreamer - relay to central
            session_id = msg.get("sessionId")
            if session_id and self._central_ws:
                await self._send_to_central(msg)

        elif msg_type == "endSession":
            session_id = msg.get("sessionId")
            if session_id:
                self._session_to_local_peer.pop(session_id, None)
                await self._send_to_central(msg)

    async def _send_to_central(self, msg: dict) -> None:
        """Send a message to the central server."""
        if self._central_ws:
            try:
                await self._central_ws.send(json.dumps(msg))
            except Exception as e:
                logger.error(f"Failed to send to central: {e}")

    async def _send_to_local(self, msg: dict) -> None:
        """Send a message to local GStreamer signaling."""
        if self._local_ws:
            try:
                await self._local_ws.send(json.dumps(msg))
            except Exception as e:
                logger.error(f"Failed to send to local: {e}")


# Singleton instance for integration
_relay_instance: Optional[CentralSignalingRelay] = None


def get_relay() -> Optional[CentralSignalingRelay]:
    """Get the global relay instance."""
    return _relay_instance


async def start_central_relay(
    hf_token: Optional[str] = None,
    robot_name: str = "reachymini",
    central_uri: str = CENTRAL_SIGNALING_SERVER,
) -> CentralSignalingRelay:
    """Start the central signaling relay.

    Args:
        hf_token: HuggingFace token for authentication
        robot_name: Name to register as producer
        central_uri: Central server URI

    Returns:
        The relay instance

    """
    global _relay_instance

    if _relay_instance is not None:
        return _relay_instance

    # Try to get HF token if not provided
    if hf_token is None:
        try:
            from huggingface_hub import get_token
            hf_token = get_token()
        except Exception:
            pass

    _relay_instance = CentralSignalingRelay(
        central_uri=central_uri,
        hf_token=hf_token,
        robot_name=robot_name,
    )
    await _relay_instance.start()
    return _relay_instance


async def stop_central_relay() -> None:
    """Stop the central signaling relay."""
    global _relay_instance

    if _relay_instance:
        await _relay_instance.stop()
        _relay_instance = None
