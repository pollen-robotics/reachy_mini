"""Central signaling relay for WebRTC.

Connects to a central signaling server via HTTP/SSE and relays messages to/from
the local GStreamer webrtcsink signaling server.
"""

import asyncio
import json
import logging
from typing import Optional

import aiohttp
import websockets
from websockets.asyncio.client import ClientConnection

logger = logging.getLogger(__name__)

# Central signaling server URL
CENTRAL_SIGNALING_SERVER = "https://cduss-reachy-mini-central.hf.space"
LOCAL_GSTREAMER_SIGNALING = "ws://127.0.0.1:8443"

# Reconnection settings
RECONNECT_INTERVAL = 5.0  # seconds


class CentralSignalingRelay:
    """Relay signaling messages between central server (HTTP/SSE) and local GStreamer (WebSocket)."""

    def __init__(
        self,
        central_uri: str = CENTRAL_SIGNALING_SERVER,
        local_uri: str = LOCAL_GSTREAMER_SIGNALING,
        hf_token: Optional[str] = None,
        robot_name: str = "reachymini",
    ):
        """Initialize the relay.

        Args:
            central_uri: HTTP URI of central signaling server
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
        self._local_ws: Optional[ClientConnection] = None
        self._central_peer_id: Optional[str] = None
        self._local_peer_id: Optional[str] = None
        self._http_session: Optional[aiohttp.ClientSession] = None

        # Map central session IDs to client peer IDs
        self._session_to_local_peer: dict[str, str] = {}
        self._local_producer_id: Optional[str] = None

        # Session ID mapping between central and local
        self._pending_central_sessions: list[str] = []  # Central sessions waiting for local session
        self._local_to_central_session: dict[str, str] = {}  # local_session_id -> central_session_id
        self._central_to_local_session: dict[str, str] = {}  # central_session_id -> local_session_id

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
        """Close all connections."""
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

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

        # Create HTTP session for central server
        self._http_session = aiohttp.ClientSession()

        # Connect to local GStreamer signaling (WebSocket)
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
            await self._http_session.close()
            raise

        try:
            # Start both handlers
            await asyncio.gather(
                self._handle_central_sse(),
                self._handle_local_messages(),
            )
        finally:
            await self._close_connections()

    async def _handle_central_sse(self) -> None:
        """Handle SSE events from central server."""
        events_url = f"{self.central_uri}/events?token={self.hf_token}"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        logger.info(f"Connecting to central SSE: {self.central_uri}/events")

        try:
            async with self._http_session.get(events_url, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Failed to connect to central SSE: {response.status}")
                    return

                logger.info("Connected to central signaling server (SSE)")

                async for line in response.content:
                    if not self._running:
                        break

                    line = line.decode('utf-8').strip()

                    if line.startswith('data:'):
                        data = line[5:].strip()
                        if data:
                            try:
                                msg = json.loads(data)
                                await self._process_central_message(msg)
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON from central: {data[:100]}")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error in central SSE: {e}")

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

    async def _send_to_central(self, msg: dict) -> None:
        """Send a message to the central server via HTTP POST."""
        if not self._http_session or not self.hf_token:
            return

        send_url = f"{self.central_uri}/send?token={self.hf_token}"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        try:
            async with self._http_session.post(send_url, json=msg, headers=headers) as response:
                if response.status != 200:
                    logger.warning(f"Failed to send to central: {response.status}")
        except Exception as e:
            logger.error(f"Error sending to central: {e}")

    async def _send_to_local(self, msg: dict) -> None:
        """Send a message to local GStreamer signaling."""
        if self._local_ws:
            try:
                await self._local_ws.send(json.dumps(msg))
            except Exception as e:
                logger.error(f"Failed to send to local: {e}")

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

        elif msg_type == "list":
            # Ignore list messages - we're a producer
            pass

        elif msg_type == "startSession":
            # A client wants to connect - forward to local GStreamer
            client_peer_id = msg.get("peerId")
            session_id = msg.get("sessionId")
            logger.info(f"Session request from client: {client_peer_id}, session: {session_id}")

            # Store session mapping
            if session_id:
                self._session_to_local_peer[session_id] = client_peer_id
                self._pending_central_sessions.append(session_id)

            # Request list of local producers to start session
            await self._send_to_local({"type": "list"})

        elif msg_type == "peer":
            # SDP/ICE from client - relay to local GStreamer
            central_session_id = msg.get("sessionId")
            if central_session_id and self._local_ws:
                # Translate central session ID to local session ID
                local_session_id = self._central_to_local_session.get(central_session_id)
                if not local_session_id:
                    logger.warning(f"No local session mapping for central session: {central_session_id}")
                    return

                local_msg = {
                    "type": "peer",
                    "sessionId": local_session_id,
                }
                if "sdp" in msg:
                    local_msg["sdp"] = msg["sdp"]
                if "ice" in msg:
                    local_msg["ice"] = msg["ice"]

                logger.debug(f"Relaying peer msg from central {central_session_id} to local {local_session_id}")
                await self._send_to_local(local_msg)

        elif msg_type == "endSession":
            central_session_id = msg.get("sessionId")
            if central_session_id:
                self._session_to_local_peer.pop(central_session_id, None)
                # Translate and forward to local
                local_session_id = self._central_to_local_session.pop(central_session_id, None)
                if local_session_id:
                    self._local_to_central_session.pop(local_session_id, None)
                    await self._send_to_local({"type": "endSession", "sessionId": local_session_id})

        elif msg_type == "peerStatusChanged":
            # Another peer changed status - ignore for producers
            pass

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
                self._local_producer_id = producers[0].get("id")
                logger.debug(f"Local producer found: {self._local_producer_id}")

                # If we have pending sessions, start them
                for session_id in list(self._session_to_local_peer.keys()):
                    await self._send_to_local({
                        "type": "startSession",
                        "peerId": self._local_producer_id,
                    })

        elif msg_type == "peerStatusChanged":
            peer_id = msg.get("peerId")
            roles = msg.get("roles", [])
            if "producer" in roles:
                self._local_producer_id = peer_id
                logger.debug(f"Local producer registered: {peer_id}")

        elif msg_type == "sessionStarted":
            local_session_id = msg.get("sessionId")
            logger.info(f"Local session started: {local_session_id}")

            # Map local session ID to the pending central session ID
            if self._pending_central_sessions:
                central_session_id = self._pending_central_sessions.pop(0)
                self._local_to_central_session[local_session_id] = central_session_id
                self._central_to_local_session[central_session_id] = local_session_id
                logger.info(f"Session mapping: local {local_session_id} <-> central {central_session_id}")

        elif msg_type == "peer":
            # SDP/ICE from local GStreamer - relay to central
            local_session_id = msg.get("sessionId")
            if local_session_id:
                # Translate local session ID to central session ID
                central_session_id = self._local_to_central_session.get(local_session_id)
                if not central_session_id:
                    logger.warning(f"No central session mapping for local session: {local_session_id}")
                    return

                # Build message with translated session ID
                central_msg = {
                    "type": "peer",
                    "sessionId": central_session_id,
                }
                if "sdp" in msg:
                    central_msg["sdp"] = msg["sdp"]
                if "ice" in msg:
                    central_msg["ice"] = msg["ice"]

                logger.debug(f"Relaying peer msg from local {local_session_id} to central {central_session_id}")
                await self._send_to_central(central_msg)

        elif msg_type == "endSession":
            local_session_id = msg.get("sessionId")
            if local_session_id:
                # Translate and forward to central
                central_session_id = self._local_to_central_session.pop(local_session_id, None)
                if central_session_id:
                    self._central_to_local_session.pop(central_session_id, None)
                    self._session_to_local_peer.pop(central_session_id, None)
                    await self._send_to_central({"type": "endSession", "sessionId": central_session_id})


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
