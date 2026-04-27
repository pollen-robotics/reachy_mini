"""Central signaling relay for WebRTC.

Connects to a central signaling server via HTTP/SSE and relays messages to/from
the local GStreamer webrtcsink signaling server.

The relay automatically:
- Reconnects on connection failures
- Refreshes the HF token on reconnection attempts
- Responds to token updates (login/logout) without restart
"""

import asyncio
import json
import logging
import threading
import time
from enum import Enum
from typing import Any, Callable, Optional

import aiohttp
import websockets
from websockets.asyncio.client import ClientConnection

from reachy_mini.daemon.robot_app_lock import RobotAppLock

logger = logging.getLogger(__name__)


class RelayState(Enum):
    """Connection state of the central signaling relay."""

    STOPPED = "stopped"
    WAITING_FOR_TOKEN = "waiting_for_token"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


# Central signaling server URL
CENTRAL_SIGNALING_SERVER = "https://cduss-reachy-mini-central.hf.space"
LOCAL_GSTREAMER_SIGNALING = "ws://127.0.0.1:8443"

# Reconnection settings
RECONNECT_INTERVAL = 5.0  # seconds
TOKEN_CHECK_INTERVAL = 30.0  # seconds - how often to check for token when not connected
LOCAL_WS_CONNECT_TIMEOUT = 5.0  # seconds - timeout for local websocket connection
LOCAL_WS_WELCOME_TIMEOUT = (
    3.0  # seconds - timeout waiting for welcome message from local
)
SSE_READ_TIMEOUT = (
    60.0  # seconds - timeout for reading from SSE stream (should receive keepalive)
)
CONNECTION_WATCHDOG_INTERVAL = 30.0  # seconds - how often to check connection health
CONNECTION_STALE_THRESHOLD = 90.0  # seconds - consider connection stale if no activity


class CentralSignalingRelay:
    """Relay signaling messages between central server (HTTP/SSE) and local GStreamer (WebSocket).

    This class maintains connections to both a central signaling server (for remote access)
    and the local GStreamer WebRTC signaling server, relaying messages between them.

    The relay is designed to be robust:
    - Automatically reconnects on connection failures
    - Refreshes the HF token on each reconnection attempt
    - Can be notified of token changes for immediate reconnection
    """

    def __init__(
        self,
        central_uri: str = CENTRAL_SIGNALING_SERVER,
        local_uri: str = LOCAL_GSTREAMER_SIGNALING,
        hf_token: Optional[str] = None,
        robot_name: str = "reachymini",
        on_state_change: Optional[Callable[["RelayState", Optional[str]], None]] = None,
        robot_app_lock: Optional[RobotAppLock] = None,
    ):
        """Initialize the relay.

        Args:
            central_uri: HTTP URI of central signaling server
            local_uri: WebSocket URI of local GStreamer signaling server
            hf_token: HuggingFace token for authentication (will be refreshed)
            robot_name: Name to register as producer
            on_state_change: Callback when state changes (state, message)
            robot_app_lock: Shared lock coordinating local vs remote access to
                the robot. When provided, incoming remote sessions are
                gated on this lock and a local-app acquire evicts any
                active remote session.

        """
        self.central_uri = central_uri
        self.local_uri = local_uri
        self.hf_token = hf_token
        self.robot_name = robot_name
        self._on_state_change = on_state_change
        self._robot_app_lock = robot_app_lock

        self._running = False
        self._state = RelayState.STOPPED
        self._state_message: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._thread_loop: Optional["asyncio.AbstractEventLoop"] = None
        self._local_ws: Optional[ClientConnection] = None
        self._central_peer_id: Optional[str] = None
        self._local_peer_id: Optional[str] = None
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._connection_attempts = 0

        # Event to signal token update (triggers immediate reconnection)
        self._token_updated = asyncio.Event()

        # Map central session IDs to client peer IDs
        self._session_to_local_peer: dict[str, str] = {}
        self._local_producer_id: Optional[str] = None

        # Session ID mapping between central and local
        self._pending_central_sessions: list[
            str
        ] = []  # Central sessions waiting for local session
        self._local_to_central_session: dict[
            str, str
        ] = {}  # local_session_id -> central_session_id
        self._central_to_local_session: dict[
            str, str
        ] = {}  # central_session_id -> local_session_id

        # Connection health tracking
        self._last_central_activity: float = 0.0

    @property
    def state(self) -> RelayState:
        """Get the current connection state."""
        return self._state

    @property
    def state_message(self) -> Optional[str]:
        """Get additional info about the current state."""
        return self._state_message

    def _set_state(self, state: RelayState, message: Optional[str] = None) -> None:
        """Update the connection state with logging."""
        old_state = self._state
        if old_state == state and self._state_message == message:
            return

        self._state = state
        self._state_message = message

        # Log state transition with appropriate level
        log_msg = (
            f"[Central Relay] State transition: {old_state.value} -> {state.value}"
        )
        if message:
            log_msg += f" | {message}"

        if state == RelayState.CONNECTED:
            logger.info(log_msg)
        elif state == RelayState.ERROR:
            logger.warning(log_msg)
        elif state == RelayState.WAITING_FOR_TOKEN:
            logger.info(log_msg)
        elif state == RelayState.RECONNECTING:
            logger.info(log_msg)
        elif state == RelayState.CONNECTING:
            logger.debug(log_msg)
        elif state == RelayState.STOPPED:
            logger.info(log_msg)
        else:
            logger.debug(log_msg)

        # Structured kv form: easy to grep, easy to dashboard. The
        # human-readable line above stays for compatibility with anyone
        # reading the systemd journal directly.
        from reachy_mini.daemon.app.logging_ctx import kv_log

        kv_log(
            logger,
            logging.INFO if state != RelayState.ERROR else logging.WARNING,
            "central.relay.state",
            from_=old_state.value,
            to=state.value,
            message=message,
        )

        # Notify callback if set
        if self._on_state_change:
            try:
                self._on_state_change(state, message)
            except Exception as e:
                logger.debug(f"[Central Relay] State change callback error: {e}")

    async def start(self) -> None:
        """Start the relay service."""
        if self._running:
            logger.debug("[Central Relay] start() called but already running")
            return

        logger.info("[Central Relay] Starting relay service...")
        self._running = True
        self._connection_attempts = 0
        self._token_updated.clear()
        self._set_state(RelayState.CONNECTING, "Starting relay service...")

        # Register ourselves as the remote-eviction handler for the lock.
        # When AppManager acquires the lock for a local Python app, this
        # coroutine is invoked on the main asyncio loop and schedules the
        # actual tear-down on the relay's own thread loop.
        if self._robot_app_lock is not None:
            self._robot_app_lock.set_remote_eviction_handler(
                self._handle_remote_eviction
            )

        # Run the relay in its own thread with a dedicated event loop.
        # This is necessary because the caller (daemon.start) may run in a temporary
        # event loop that gets destroyed when the HTTP request handler completes.
        self._thread = threading.Thread(target=self._run_in_thread, daemon=True)
        self._thread.start()
        logger.info(f"[Central Relay] Relay thread started: {self._thread.name}")

    async def _handle_remote_eviction(self) -> None:
        """Cross-thread entry point for the lock's remote-eviction callback.

        Runs on the main asyncio loop (caller of ``acquire_local_evicting_remote``)
        but dispatches the actual work onto the relay's thread loop.
        """
        if self._thread_loop is None or not self._thread_loop.is_running():
            logger.debug(
                "[Central Relay] Eviction requested but relay loop not running; nothing to do"
            )
            return

        fut = asyncio.run_coroutine_threadsafe(
            self._tear_down_active_sessions(reason="local_app_started"),
            self._thread_loop,
        )
        # Wait for the tear-down to complete so AppManager knows the remote
        # peer has been notified before the local app starts up.
        try:
            await asyncio.wrap_future(fut)
        except Exception:
            logger.warning(
                "[Central Relay] Remote eviction tear-down raised", exc_info=True
            )

    async def _tear_down_active_sessions(self, reason: str) -> None:
        """End every active/pending remote session and notify both sides.

        Runs on the relay's thread event loop.
        """
        # Notify central so it clears its own session_id on the producer
        # and sends endSession to the remote consumer.
        for central_session_id in list(self._central_to_local_session.keys()):
            logger.info(
                "[Central Relay] Tearing down central session %s (reason=%s)",
                central_session_id,
                reason,
            )
            try:
                await self._send_to_central(
                    {
                        "type": "endSession",
                        "sessionId": central_session_id,
                        "reason": reason,
                    }
                )
            except Exception:
                logger.warning(
                    "[Central Relay] Failed to notify central of session teardown",
                    exc_info=True,
                )

        # Notify local GStreamer so it closes its RTCPeerConnection.
        for local_session_id in list(self._local_to_central_session.keys()):
            try:
                await self._send_to_local(
                    {"type": "endSession", "sessionId": local_session_id}
                )
            except Exception:
                logger.warning(
                    "[Central Relay] Failed to notify local of session teardown",
                    exc_info=True,
                )

        # Clear all session bookkeeping.
        self._pending_central_sessions.clear()
        self._local_to_central_session.clear()
        self._central_to_local_session.clear()
        self._session_to_local_peer.clear()

    def _run_in_thread(self) -> None:
        """Run the relay loop in a dedicated thread with its own event loop."""
        logger.info("[Central Relay] Thread starting, creating event loop...")
        self._thread_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._thread_loop)
        try:
            self._thread_loop.run_until_complete(self._run_loop())
        except Exception as e:
            logger.error(f"[Central Relay] Thread event loop error: {e}")
        finally:
            logger.info("[Central Relay] Thread event loop finished")
            self._thread_loop.close()
            self._thread_loop = None

    async def stop(self) -> None:
        """Stop the relay service."""
        logger.info("[Central Relay] Stopping relay service...")
        self._running = False

        # Wake up any waiting in the thread's event loop
        if self._thread_loop and self._thread_loop.is_running():
            self._thread_loop.call_soon_threadsafe(self._token_updated.set)

        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("[Central Relay] Thread did not stop within timeout")

        self._thread = None

        # Unregister from the lock and release any hold we may still have.
        if self._robot_app_lock is not None:
            self._robot_app_lock.set_remote_eviction_handler(None)
            self._robot_app_lock.release_remote()

        self._set_state(RelayState.STOPPED, "Relay stopped")

    async def update_token(self, new_token: Optional[str]) -> None:
        """Update the HF token and trigger reconnection if needed.

        This method should be called when the user logs in or out of
        HuggingFace. It will:
        - Update the stored token
        - Close existing connections
        - Trigger immediate reconnection attempt

        Args:
            new_token: The new HF token, or None if logged out

        """
        old_token = self.hf_token
        self.hf_token = new_token

        if old_token == new_token:
            logger.debug("[Central Relay] Token unchanged, no action needed")
            return

        await self._reconnect_now(new_token, reason="HF token updated, reconnecting...")

    async def force_reconnect(self) -> None:
        """Drop the current connection and reconnect with the stored token.

        Unlike ``update_token``, this path is NOT guarded by a token
        equality check - it always tears down the SSE and reconnects.

        Intended as a recovery handle for split-brain states where the
        relay thinks it is connected but central no longer lists this
        robot as a producer (see ``POST /api/hf-auth/refresh-relay``).
        """
        await self._reconnect_now(self.hf_token, reason="Forced relay reconnect")

    async def _reconnect_now(self, token: Optional[str], reason: str) -> None:
        """Shared core of token-change and force-reconnect paths.

        Transitions the relay into the right state and signals the run
        loop to tear down the current connection and try connecting
        again. Safe to call from any thread - if we have a running
        thread loop we schedule the close/set there, otherwise we set
        the event directly (covers the case where the relay has not
        started its background thread yet).
        """
        if token:
            self._set_state(RelayState.RECONNECTING, reason)
            self._connection_attempts = 0
        else:
            self._set_state(RelayState.WAITING_FOR_TOKEN, "Logged out from HuggingFace")

        if self._thread_loop and self._thread_loop.is_running():

            async def _reconnect() -> None:
                await self._close_connections()
                self._token_updated.set()

            asyncio.run_coroutine_threadsafe(_reconnect(), self._thread_loop)
        else:
            self._token_updated.set()

    def _refresh_token(self) -> Optional[str]:
        """Refresh the HF token from huggingface_hub.

        Returns:
            The current HF token, or None if not available

        """
        try:
            from huggingface_hub import get_token

            token = get_token()
            if token != self.hf_token:
                if token:
                    logger.info("[Central Relay] HF token detected (user logged in)")
                else:
                    logger.debug("[Central Relay] No HF token available")
                self.hf_token = token
            return token
        except Exception as e:
            logger.debug(f"[Central Relay] Could not get HF token: {e}")
            return self.hf_token

    async def _close_connections(self) -> None:
        """Close all connections."""
        if self._http_session:
            try:
                await self._http_session.close()
            except Exception:
                pass
            self._http_session = None

        if self._local_ws:
            try:
                await self._local_ws.close()
            except Exception:
                pass
            self._local_ws = None

        # Clear session state
        self._central_peer_id = None
        self._local_peer_id = None
        self._session_to_local_peer.clear()
        self._pending_central_sessions.clear()
        self._local_to_central_session.clear()
        self._central_to_local_session.clear()

        # Release any remote hold on the robot lock. Idempotent: no-op if
        # we weren't holding it (e.g. local app currently has the robot).
        if self._robot_app_lock is not None:
            self._robot_app_lock.release_remote()

    async def _run_loop(self) -> None:
        """Maintain connections and relay messages."""
        logger.info("[Central Relay] _run_loop started")

        try:
            # Small yield to allow event loop to process other events
            await asyncio.sleep(0)
            logger.info("[Central Relay] Starting connection attempts")

            while self._running:
                had_exception = False
                try:
                    await self._connect_and_relay()
                except asyncio.CancelledError:
                    # CancelledError at this layer has two possible sources:
                    #
                    # 1. `stop()` flipped `self._running` to False. This is the
                    #    shutdown path and we must re-raise so the thread exits.
                    # 2. An in-flight `_close_connections()` (triggered by
                    #    `force_reconnect` / `update_token`) cancelled a task
                    #    that aiohttp was holding the cancellation for - e.g.
                    #    a `session.get(...)` wedged inside `_resolve_host` on
                    #    a flaky network. aiohttp propagates that cancellation
                    #    up through `_handle_central_sse`, which lands here.
                    #    In that case we very much want to stay in the loop and
                    #    reconnect - killing the thread here means every
                    #    subsequent `/refresh-relay` POST is a no-op because
                    #    there's no loop left to service the token_updated
                    #    event.
                    #
                    # `self._running` is our authoritative signal for (1). If
                    # it's still True, treat the cancellation as a reconnect
                    # request and loop around.
                    if not self._running:
                        logger.info(
                            "[Central Relay] _run_loop cancelled (stop requested)"
                        )
                        raise
                    logger.info(
                        "[Central Relay] Connect attempt cancelled mid-flight "
                        "(likely from force_reconnect / token update); restarting loop"
                    )
                    had_exception = True
                    self._set_state(
                        RelayState.RECONNECTING,
                        "Restarting after cancelled connect",
                    )
                except Exception as e:
                    logger.warning(
                        f"[Central Relay] Connection attempt failed with exception: {type(e).__name__}: {e}"
                    )
                    had_exception = True
                    self._connection_attempts += 1
                    if self._connection_attempts <= 3:
                        self._set_state(
                            RelayState.RECONNECTING, f"Connection failed: {e}"
                        )
                    else:
                        self._set_state(
                            RelayState.ERROR,
                            f"Connection failed after {self._connection_attempts} attempts: {e}",
                        )

                if self._running and had_exception:
                    # Only wait after connection failures, not after normal returns
                    # (e.g., when token update triggered reconnection)
                    self._token_updated.clear()
                    try:
                        await asyncio.wait_for(
                            self._token_updated.wait(), timeout=RECONNECT_INTERVAL
                        )
                    except asyncio.TimeoutError:
                        pass

        except asyncio.CancelledError:
            logger.info("[Central Relay] Run loop cancelled (stop requested)")
            raise
        except Exception as e:
            logger.error(f"[Central Relay] Unexpected error in run loop: {e}")
            raise

    async def _connect_and_relay(self) -> None:
        """Connect to both servers and relay messages."""
        logger.info("[Central Relay] _connect_and_relay() starting")

        # Always refresh the token on each connection attempt
        self._refresh_token()

        if not self.hf_token:
            self._set_state(
                RelayState.WAITING_FOR_TOKEN,
                "Login to HuggingFace to enable remote access",
            )
            # Wait longer when no token - user needs to log in
            self._token_updated.clear()
            try:
                await asyncio.wait_for(
                    self._token_updated.wait(), timeout=TOKEN_CHECK_INTERVAL
                )
                logger.info(
                    "[Central Relay] Token update received while waiting, will attempt connection"
                )
            except asyncio.TimeoutError:
                logger.debug("[Central Relay] Token check timeout, will re-check")
            return

        # Create HTTP session for central server
        self._http_session = aiohttp.ClientSession()

        # Connect to local GStreamer signaling (WebSocket) with timeout
        self._set_state(RelayState.CONNECTING, "Connecting to local WebRTC...")
        logger.info(
            f"[Central Relay] Attempting to connect to local websocket: {self.local_uri}"
        )
        try:
            self._local_ws = await asyncio.wait_for(
                websockets.connect(
                    self.local_uri,
                    ping_interval=None,
                    ping_timeout=None,
                ),
                timeout=LOCAL_WS_CONNECT_TIMEOUT,
            )
            logger.info("[Central Relay] Local websocket connection established")
        except asyncio.TimeoutError:
            logger.error(
                f"[Central Relay] Local WebRTC connection timeout after {LOCAL_WS_CONNECT_TIMEOUT}s"
            )
            self._set_state(
                RelayState.ERROR,
                f"Local WebRTC connection timeout after {LOCAL_WS_CONNECT_TIMEOUT}s",
            )
            await self._http_session.close()
            self._http_session = None
            raise
        except Exception as e:
            logger.error(f"[Central Relay] Local WebRTC connection failed: {e}")
            self._set_state(RelayState.ERROR, f"Local WebRTC unavailable: {e}")
            await self._http_session.close()
            self._http_session = None
            raise

        # Wait for welcome message from local websocket to verify connection is working
        self._local_welcome_received = asyncio.Event()
        logger.info(
            "[Central Relay] Waiting for welcome message from local websocket..."
        )
        try:
            # Start reading local messages in background to receive welcome
            local_task = asyncio.create_task(self._handle_local_messages())
            logger.info("[Central Relay] Local message handler task started")

            # Wait for welcome with timeout
            try:
                await asyncio.wait_for(
                    self._local_welcome_received.wait(),
                    timeout=LOCAL_WS_WELCOME_TIMEOUT,
                )
                logger.info(
                    "[Central Relay] Local WebRTC connection verified (welcome received)"
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"[Central Relay] Welcome message timeout after {LOCAL_WS_WELCOME_TIMEOUT}s"
                )
                local_task.cancel()
                try:
                    await local_task
                except asyncio.CancelledError:
                    pass
                self._set_state(
                    RelayState.ERROR,
                    f"Local WebRTC did not respond within {LOCAL_WS_WELCOME_TIMEOUT}s",
                )
                raise

            # Now connect to central server and run all handlers
            # Use wait with FIRST_COMPLETED so we can reconnect if any handler exits
            self._set_state(RelayState.CONNECTING, "Connecting to central server...")
            central_task = asyncio.create_task(self._handle_central_sse())
            token_task = asyncio.create_task(self._watch_for_token_update())
            tasks = {central_task, local_task, token_task}

            try:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )

                # Log which task finished first
                for task in done:
                    if task == central_task:
                        logger.info(
                            "[Central Relay] Central SSE handler exited, will reconnect"
                        )
                    elif task == local_task:
                        logger.info(
                            "[Central Relay] Local WebSocket handler exited, will reconnect"
                        )
                    elif task == token_task:
                        logger.info("[Central Relay] Token update triggered reconnect")

                    # Check for exceptions
                    if task.exception():
                        logger.warning(
                            f"[Central Relay] Task {task.get_name()} raised: {task.exception()}"
                        )

                # Cancel remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Update state to show we're reconnecting (unless already set)
                if self._state == RelayState.CONNECTED:
                    self._set_state(
                        RelayState.RECONNECTING, "Connection lost, reconnecting..."
                    )
            except asyncio.CancelledError:
                # Cancel all tasks if we're cancelled
                for task in tasks:
                    task.cancel()
                raise
        finally:
            await self._close_connections()

    async def _watch_for_token_update(self) -> None:
        """Watch for token updates and close connections to trigger reconnect."""
        await self._token_updated.wait()
        logger.debug(
            "[Central Relay] Token update signal received, closing connections"
        )
        await self._close_connections()

    async def _handle_central_sse(self) -> None:
        """Handle SSE events from central server."""
        if not self._http_session:
            return

        # Token goes in the Authorization header, never in the URL —
        # keeps it out of HF Space access logs and intermediate proxies.
        events_url = f"{self.central_uri}/events"
        headers = {"Authorization": f"Bearer {self.hf_token}"}

        try:
            # Use timeout for the initial connection
            timeout = aiohttp.ClientTimeout(
                total=None, connect=10, sock_read=SSE_READ_TIMEOUT
            )
            async with self._http_session.get(
                events_url, headers=headers, timeout=timeout
            ) as response:
                if response.status == 401:
                    self._set_state(
                        RelayState.ERROR, "Authentication failed - token may be invalid"
                    )
                    return
                elif response.status != 200:
                    self._set_state(
                        RelayState.ERROR,
                        f"Central server returned HTTP {response.status}",
                    )
                    return

                # Connection successful - will set CONNECTED after welcome message
                self._connection_attempts = 0
                self._last_central_activity = time.time()

                # Read lines with timeout to detect dead connections
                while self._running:
                    try:
                        line = await asyncio.wait_for(
                            response.content.readline(), timeout=SSE_READ_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"[Central Relay] SSE read timeout after {SSE_READ_TIMEOUT}s - connection may be dead"
                        )
                        self._set_state(
                            RelayState.RECONNECTING,
                            "Connection timeout, reconnecting...",
                        )
                        return

                    if not line:
                        # Empty line means connection closed
                        logger.info("[Central Relay] SSE connection closed by server")
                        return

                    self._last_central_activity = time.time()
                    line_str = line.decode("utf-8").strip()

                    if line_str.startswith("data:"):
                        data = line_str[5:].strip()
                        if data:
                            try:
                                msg = json.loads(data)
                                await self._process_central_message(msg)
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"[Central Relay] Invalid JSON from central: {data[:100]}"
                                )

        except asyncio.CancelledError:
            raise
        except aiohttp.ClientError as e:
            self._set_state(RelayState.ERROR, f"Central server unreachable: {e}")
        except Exception as e:
            logger.error(f"[Central Relay] Error in central SSE: {e}")
        finally:
            # Clean up all sessions when central connection drops
            if self._local_to_central_session:
                logger.info(
                    f"[Central Relay] Central connection lost, cleaning up {len(self._local_to_central_session)} sessions"
                )
                for local_session_id in list(self._local_to_central_session.keys()):
                    await self._send_to_local(
                        {"type": "endSession", "sessionId": local_session_id}
                    )

    async def _handle_local_messages(self) -> None:
        """Handle messages from local GStreamer signaling."""
        if not self._local_ws:
            return

        try:
            async for message in self._local_ws:
                try:
                    message_str = (
                        message if isinstance(message, str) else message.decode("utf-8")
                    )
                    msg = json.loads(message_str)
                    await self._process_local_message(msg)
                except json.JSONDecodeError:
                    logger.warning(
                        f"[Central Relay] Invalid JSON from local GStreamer: {str(message)[:100]}"
                    )
        except websockets.ConnectionClosed:
            logger.info("[Central Relay] Local GStreamer WebSocket connection closed")
        except Exception as e:
            logger.error(
                f"[Central Relay] Error handling local GStreamer messages: {e}"
            )

    async def _send_to_central(self, msg: dict[str, Any]) -> None:
        """Send a message to the central server via HTTP POST."""
        msg_type = msg.get("type", "?")
        if not self._http_session or not self.hf_token:
            logger.warning(
                "[Central Relay] _send_to_central skipped (type=%s, http_session=%s, hf_token=%s)",
                msg_type,
                bool(self._http_session),
                bool(self.hf_token),
            )
            return

        # Token goes in the Authorization header only, not the URL.
        send_url = f"{self.central_uri}/send"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        try:
            async with self._http_session.post(
                send_url, json=msg, headers=headers
            ) as response:
                if response.status != 200:
                    body = ""
                    try:
                        body = (await response.text())[:300]
                    except Exception:
                        pass
                    logger.warning(
                        "[Central Relay] _send_to_central FAILED type=%s HTTP %s body=%r",
                        msg_type,
                        response.status,
                        body,
                    )
                else:
                    logger.info("[Central Relay] _send_to_central OK type=%s", msg_type)
        except Exception as e:
            logger.error(
                "[Central Relay] _send_to_central exception type=%s err=%s",
                msg_type,
                e,
            )

    async def _send_to_local(self, msg: dict[str, Any]) -> None:
        """Send a message to local GStreamer signaling."""
        if self._local_ws:
            try:
                await self._local_ws.send(json.dumps(msg))
            except Exception as e:
                logger.error(
                    f"[Central Relay] Failed to send message to local GStreamer: {e}"
                )

    async def _process_central_message(self, msg: dict[str, Any]) -> None:
        """Process a message from the central server."""
        msg_type = msg.get("type", "")
        logger.debug(f"[Central Relay] Received from central server: type={msg_type}")

        if msg_type == "welcome":
            # Received our peer ID from central server
            self._central_peer_id = msg.get("peerId")
            logger.info(
                "[Central Relay] central welcome received peer_id=%s; registering as producer name=%r",
                self._central_peer_id,
                self.robot_name,
            )

            # Register as producer FIRST, then flip to CONNECTED. If we
            # set CONNECTED before producer registration, observers (UI,
            # mobile app, /relay-status pollers) can see "connected" while
            # central does not yet know we are a producer for this user,
            # which produces the desync described in /refresh-relay's
            # docstring.
            await self._send_to_central(
                {
                    "type": "setPeerStatus",
                    "roles": ["producer"],
                    "meta": {"name": self.robot_name},
                }
            )

            self._set_state(
                RelayState.CONNECTED, f"Remote access enabled as '{self.robot_name}'"
            )

        elif msg_type == "list":
            # Ignore list messages - we're a producer
            pass

        elif msg_type == "startSession":
            # A client wants to connect - forward to local GStreamer
            client_peer_id: str = msg.get("peerId", "")
            session_id: Optional[str] = msg.get("sessionId")
            logger.info(
                f"[Central Relay] Received session request from remote client peer_id={client_peer_id} session_id={session_id}"
            )

            # Safety net: the central server is supposed to gate concurrent sessions,
            # but if one slips through (e.g. older central without the gate), enforce
            # single-session-at-a-time here so we never run two clients against one robot.
            if self._central_to_local_session or self._pending_central_sessions:
                logger.warning(
                    f"[Central Relay] Rejecting session {session_id}: a session is already active/pending"
                )
                if session_id:
                    await self._send_to_central(
                        {
                            "type": "endSession",
                            "sessionId": session_id,
                            "reason": "robot_busy_local",
                        }
                    )
                return

            # Gate on the robot lock: if a local Python app is running, the
            # lock will refuse our acquire. We also acquire proactively here
            # so a concurrent local-app start can't sneak in between the
            # check and the session handoff to local GStreamer.
            if self._robot_app_lock is not None:
                # holder_name is generic because central already tracks the
                # real consumer app name (via setPeerStatus meta) for its
                # own rejection messages; the daemon-side lock just needs
                # to know that *something* remote holds it.
                if not self._robot_app_lock.try_acquire_remote("remote"):
                    logger.warning(
                        f"[Central Relay] Rejecting session {session_id}: robot lock is held locally"
                    )
                    if session_id:
                        await self._send_to_central(
                            {
                                "type": "endSession",
                                "sessionId": session_id,
                                "reason": "robot_busy_local_app",
                            }
                        )
                    return

            # Store session mapping
            if session_id:
                # Check if we already have this session (duplicate request)
                if session_id in self._session_to_local_peer:
                    logger.warning(
                        f"[Central Relay] Duplicate session request for session_id={session_id}, ignoring"
                    )
                    return

                self._session_to_local_peer[session_id] = client_peer_id
                self._pending_central_sessions.append(session_id)
                logger.info(
                    f"[Central Relay] Pending sessions: {len(self._pending_central_sessions)}, tracked sessions: {len(self._session_to_local_peer)}"
                )

            # Request list of local producers to start session
            await self._send_to_local({"type": "list"})

        elif msg_type == "peer":
            # SDP/ICE from client - relay to local GStreamer
            central_session_id = msg.get("sessionId")
            if central_session_id and self._local_ws:
                # Translate central session ID to local session ID
                local_session_id = self._central_to_local_session.get(
                    central_session_id
                )
                if not local_session_id:
                    logger.warning(
                        f"[Central Relay] No local session mapping found for central_session_id={central_session_id}"
                    )
                    return

                local_msg = {
                    "type": "peer",
                    "sessionId": local_session_id,
                }
                if "sdp" in msg:
                    local_msg["sdp"] = msg["sdp"]
                if "ice" in msg:
                    local_msg["ice"] = msg["ice"]

                logger.debug(
                    f"[Central Relay] Relaying peer message: central_session={central_session_id} -> local_session={local_session_id}"
                )
                await self._send_to_local(local_msg)

        elif msg_type == "endSession":
            central_session_id = msg.get("sessionId")
            if central_session_id:
                logger.info(
                    f"[Central Relay] Session ended from central: central_session_id={central_session_id}"
                )
                self._session_to_local_peer.pop(central_session_id, None)
                # Also remove from pending if it never got started
                if central_session_id in self._pending_central_sessions:
                    self._pending_central_sessions.remove(central_session_id)
                # Translate and forward to local
                local_session_id = self._central_to_local_session.pop(
                    central_session_id, None
                )
                if local_session_id:
                    self._local_to_central_session.pop(local_session_id, None)
                    logger.info(
                        f"[Central Relay] Forwarding endSession to local: local_session_id={local_session_id}"
                    )
                    await self._send_to_local(
                        {"type": "endSession", "sessionId": local_session_id}
                    )
                logger.info(
                    f"[Central Relay] After cleanup - pending: {len(self._pending_central_sessions)}, active: {len(self._central_to_local_session)}"
                )
                # If no sessions remain, release the robot lock.
                if (
                    self._robot_app_lock is not None
                    and not self._central_to_local_session
                    and not self._pending_central_sessions
                ):
                    self._robot_app_lock.release_remote()

        elif msg_type == "peerStatusChanged":
            # Another peer changed status - ignore for producers
            pass

    async def _process_local_message(self, msg: dict[str, Any]) -> None:
        """Process a message from local GStreamer signaling."""
        msg_type = msg.get("type", "")
        logger.debug(f"[Central Relay] Received from local GStreamer: type={msg_type}")

        if msg_type == "welcome":
            # Received our peer ID from local GStreamer
            self._local_peer_id = msg.get("peerId")
            logger.info(
                f"[Central Relay] Connected to local GStreamer signaling server peer_id={self._local_peer_id}"
            )

            # Signal that local connection is verified
            if hasattr(self, "_local_welcome_received"):
                self._local_welcome_received.set()

            # Register as listener to receive producer announcements
            await self._send_to_local(
                {
                    "type": "setPeerStatus",
                    "roles": ["listener"],
                    "meta": {"name": "central-relay"},
                }
            )

        elif msg_type == "list":
            # List of local producers
            producers = msg.get("producers", [])
            if producers:
                self._local_producer_id = producers[0].get("id")
                logger.debug(
                    f"[Central Relay] Local GStreamer producer found: producer_id={self._local_producer_id}"
                )

                # Only start sessions for PENDING requests, not all tracked sessions
                for central_session_id in list(self._pending_central_sessions):
                    logger.info(
                        f"[Central Relay] Starting local session for pending central_session={central_session_id}"
                    )
                    await self._send_to_local(
                        {
                            "type": "startSession",
                            "peerId": self._local_producer_id,
                        }
                    )

        elif msg_type == "peerStatusChanged":
            peer_id = msg.get("peerId")
            roles = msg.get("roles", [])
            if "producer" in roles:
                self._local_producer_id = peer_id
                logger.debug(
                    f"[Central Relay] Local GStreamer producer registered: producer_id={peer_id}"
                )

        elif msg_type == "sessionStarted":
            local_session_id: Optional[str] = msg.get("sessionId")
            logger.info(
                f"[Central Relay] Local GStreamer session started: local_session_id={local_session_id}"
            )

            # Map local session ID to the pending central session ID
            if self._pending_central_sessions and local_session_id:
                central_session_id = self._pending_central_sessions.pop(0)
                self._local_to_central_session[local_session_id] = central_session_id
                self._central_to_local_session[central_session_id] = local_session_id
                logger.info(
                    f"[Central Relay] Session mapping established: local_session={local_session_id} <-> central_session={central_session_id}"
                )

        elif msg_type == "peer":
            # SDP/ICE from local GStreamer - relay to central
            local_session_id_peer: Optional[str] = msg.get("sessionId")
            if local_session_id_peer:
                # Translate local session ID to central session ID
                central_session_id_peer: Optional[str] = (
                    self._local_to_central_session.get(local_session_id_peer)
                )
                if not central_session_id_peer:
                    logger.warning(
                        f"[Central Relay] No central session mapping found for local_session_id={local_session_id_peer}"
                    )
                    return

                # Build message with translated session ID
                central_msg: dict[str, Any] = {
                    "type": "peer",
                    "sessionId": central_session_id_peer,
                }
                if "sdp" in msg:
                    central_msg["sdp"] = msg["sdp"]
                if "ice" in msg:
                    central_msg["ice"] = msg["ice"]

                logger.debug(
                    f"[Central Relay] Relaying peer message: local_session={local_session_id_peer} -> central_session={central_session_id_peer}"
                )
                await self._send_to_central(central_msg)

        elif msg_type == "endSession":
            local_session_id_end: Optional[str] = msg.get("sessionId")
            if local_session_id_end:
                logger.info(
                    f"[Central Relay] Session ended from local: local_session_id={local_session_id_end}"
                )
                # Translate and forward to central
                central_session_id_end: Optional[str] = (
                    self._local_to_central_session.pop(local_session_id_end, None)
                )
                if central_session_id_end:
                    self._central_to_local_session.pop(central_session_id_end, None)
                    self._session_to_local_peer.pop(central_session_id_end, None)
                    logger.info(
                        f"[Central Relay] Forwarding endSession to central: central_session_id={central_session_id_end}"
                    )
                    await self._send_to_central(
                        {"type": "endSession", "sessionId": central_session_id_end}
                    )
                logger.info(
                    f"[Central Relay] After cleanup - pending: {len(self._pending_central_sessions)}, active: {len(self._central_to_local_session)}"
                )
                # If no sessions remain, release the robot lock.
                if (
                    self._robot_app_lock is not None
                    and not self._central_to_local_session
                    and not self._pending_central_sessions
                ):
                    self._robot_app_lock.release_remote()

    async def update_producer_name(self, robot_name: str) -> None:
        """Update the producer name advertised to central, live.

        Used by the daemon when the user renames the robot from the mobile
        app: we need central's view of the fleet to reflect the new label
        immediately, without forcing a full relay reconnect (which would
        churn the WebSocket and momentarily evict any active remote
        session).

        Idempotent: a no-op when the name has not changed. Safe to call
        before the relay is connected; the new value will be picked up by
        the next ``setPeerStatus`` emitted on ``welcome``.

        Must be invoked from the relay's own thread loop (or via
        ``asyncio.run_coroutine_threadsafe`` for cross-thread callers) so
        that the ``_send_to_central`` HTTP call uses the right session.
        """
        if robot_name == self.robot_name:
            return

        self.robot_name = robot_name
        logger.info(
            "[Central Relay] producer name updated to %r (state=%s)",
            robot_name,
            self._state.value,
        )

        # Only emit setPeerStatus when we're actually registered with
        # central. In any other state the next ``welcome`` will use the
        # already-updated ``self.robot_name``.
        if self._state == RelayState.CONNECTED and self._central_peer_id is not None:
            await self._send_to_central(
                {
                    "type": "setPeerStatus",
                    "roles": ["producer"],
                    "meta": {"name": self.robot_name},
                }
            )


# Singleton instance for integration
_relay_instance: Optional[CentralSignalingRelay] = None


def get_relay() -> Optional[CentralSignalingRelay]:
    """Get the global relay instance.

    Returns:
        The relay instance, or None if not started

    """
    return _relay_instance


def get_relay_status() -> dict[str, Any]:
    """Get the current status of the central relay.

    Returns:
        A dict with state, message, and is_connected fields

    """
    if _relay_instance is None:
        return {
            "state": RelayState.STOPPED.value,
            "message": "Relay not initialized",
            "is_connected": False,
        }

    return {
        "state": _relay_instance.state.value,
        "message": _relay_instance.state_message,
        "is_connected": _relay_instance.state == RelayState.CONNECTED,
    }


async def start_central_relay(
    hf_token: Optional[str] = None,
    robot_name: str = "reachymini",
    central_uri: str = CENTRAL_SIGNALING_SERVER,
    on_state_change: Optional[Callable[[RelayState, Optional[str]], None]] = None,
    robot_app_lock: Optional[RobotAppLock] = None,
) -> CentralSignalingRelay:
    """Start the central signaling relay.

    Args:
        hf_token: HuggingFace token for authentication (will auto-refresh)
        robot_name: Name to register as producer
        central_uri: Central server URI
        on_state_change: Callback when connection state changes
        robot_app_lock: Shared lock coordinating local vs remote robot access.

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
        on_state_change=on_state_change,
        robot_app_lock=robot_app_lock,
    )
    await _relay_instance.start()
    return _relay_instance


async def stop_central_relay() -> None:
    """Stop the central signaling relay."""
    global _relay_instance

    if _relay_instance:
        await _relay_instance.stop()
        _relay_instance = None


async def notify_token_change(new_token: Optional[str] = None) -> None:
    """Notify the relay of a token change (login/logout).

    This should be called from the HF auth endpoints when the user
    logs in or out. If new_token is None, it will be fetched from
    huggingface_hub.

    Args:
        new_token: The new token, or None to fetch from huggingface_hub

    """
    if _relay_instance is None:
        logger.debug("[Central Relay] No relay instance, ignoring token change")
        return

    if new_token is None:
        try:
            from huggingface_hub import get_token

            new_token = get_token()
        except Exception:
            pass

    await _relay_instance.update_token(new_token)


async def notify_force_reconnect() -> None:
    """Ask the relay to drop its SSE channel and reconnect right now.

    Unlike ``notify_token_change``, this always triggers a reconnect
    even when the stored token is unchanged. Used by the
    ``POST /api/hf-auth/refresh-relay`` endpoint to recover from
    zombie-relay states where central no longer lists the robot as a
    producer but the relay still thinks it is connected.
    """
    if _relay_instance is None:
        logger.debug("[Central Relay] No relay instance, ignoring force reconnect")
        return

    await _relay_instance.force_reconnect()


async def notify_robot_name_change(new_name: str) -> None:
    """Push a renamed robot to central without reconnecting.

    Called by ``Daemon.set_robot_name`` after the in-memory and on-disk
    state has been updated. No-op when there is no relay (Lite, no token,
    relay not started yet); the relay will pick up the new name on its
    next ``start()`` because the daemon also passes the fresh name to
    ``start_central_relay``.

    The actual ``setPeerStatus`` send happens on the relay's own thread
    loop via ``run_coroutine_threadsafe``: ``_send_to_central`` uses the
    relay-local ``aiohttp.ClientSession``, which is bound to that loop.
    """
    if _relay_instance is None:
        logger.debug(
            "[Central Relay] No relay instance, robot name change is in-memory only"
        )
        return

    loop = _relay_instance._thread_loop
    if loop is None or not loop.is_running():
        # Relay thread not up yet; just update the attribute so the next
        # ``welcome`` advertises the right name.
        _relay_instance.robot_name = new_name
        return

    fut = asyncio.run_coroutine_threadsafe(
        _relay_instance.update_producer_name(new_name), loop
    )
    try:
        # Bound the wait so a stuck relay thread cannot freeze the HTTP
        # request that triggered the rename. Failure here is non-fatal:
        # the on-disk + in-memory state is already updated, central will
        # catch up on its next reconnect / welcome.
        await asyncio.get_event_loop().run_in_executor(None, fut.result, 5.0)
    except Exception as exc:
        logger.warning(
            "[Central Relay] update_producer_name(%r) failed: %s", new_name, exc
        )
