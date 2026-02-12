"""webrtc utils functions."""

import argparse
import json
import logging
from threading import Event, Thread
from typing import Callable, Dict, Optional

from websockets.sync.client import ClientConnection, connect

logger = logging.getLogger(__name__)


def get_producer_list(host: str, port: int) -> Dict[str, Dict[str, str]]:
    """Get the list of gstreamer producers from the signalling server.

    Args:
        host (str): The hostname or IP address of the signalling server.
        port (int): The port number of the signalling server.

    Returns:
        Dict[str, Dict[str, str]]: A dictionary mapping producer IDs to their metadata dictionaries.

    """
    """Get the list of gstreamer producers from the signalling server."""
    with connect(f"ws://{host}:{port}") as websocket:
        _ = websocket.recv()  # welcome message is ignored
        message = json.dumps({"type": "list"})
        websocket.send(message)
        message = json.loads(websocket.recv())
        logging.debug(f"Received: {message}")
        if message.get("type") == "list":
            producers = {p["id"]: p["meta"] for p in message.get("producers", [])}
            return producers
        else:
            logging.warning(f"Received unknown message type: {message}.")
            return {}


def find_producer_peer_id_by_name(host: str, port: int, name: str) -> str:
    """Find the peer ID of a producer by its name.

    host (str): Host address of the producer service.
    port (int): Port number of the producer service.

    str: Peer ID of the producer (returns the first match if multiple exist).

    KeyError: If no producer with the specified name is found.
    """
    producers = get_producer_list(host=host, port=port)

    for producer_id, producer_meta in producers.items():
        if producer_meta["name"] == name:
            return producer_id

    raise KeyError(f"Producer {name} not found.")


class SignallingListener:
    """WebSocket listener that monitors peer events on the signalling server.

    Connects as a "listener" role and dispatches callbacks when producers
    appear or disappear based on ``peerStatusChanged`` messages.

    Args:
        host: Signalling server hostname.
        port: Signalling server port.
        on_producer_added: Called with (peer_id, meta_dict) when a producer appears.
        on_producer_removed: Called with (peer_id) when a producer disappears.

    """

    def __init__(  # noqa: D107
        self,
        host: str = "127.0.0.1",
        port: int = 8443,
        on_producer_added: Optional[Callable[[str, Dict[str, str]], None]] = None,
        on_producer_removed: Optional[Callable[[str], None]] = None,
        log_level: str = "INFO",
    ) -> None:
        self.logger = logging.getLogger(f"{__name__}.SignallingListener")
        self.logger.setLevel(log_level)
        self._uri = f"ws://{host}:{port}"
        self._on_producer_added = on_producer_added
        self._on_producer_removed = on_producer_removed
        self._ws: Optional[ClientConnection] = None
        self._thread: Optional[Thread] = None
        self._stop_event = Event()
        # track known producers: peer_id -> meta dict
        self._producers: Dict[str, Dict[str, str]] = {}

    def start(self) -> None:
        """Connect and start listening in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            self.logger.debug("start() called but already running, skipping")
            return
        self.logger.debug("starting listener thread")
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Disconnect and stop the listener thread."""
        self.logger.debug("stop() called, setting stop event")
        self._stop_event.set()

        if self._thread is not None:
            self.logger.debug("joining listener thread (timeout=5s)")
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                self.logger.debug("listener thread did NOT terminate within 5s")
            else:
                self.logger.debug("listener thread joined successfully")
            self._thread = None

    def _run(self) -> None:
        """Connect, register as listener, and process messages."""
        try:
            self._ws = connect(self._uri)
        except Exception as e:
            self.logger.error(f"failed to connect to {self._uri}: {e}")
            return

        try:
            # Read welcome message
            welcome_raw = self._ws.recv()
            welcome = json.loads(welcome_raw)
            my_peer_id = welcome.get("peerId", "")
            self.logger.debug(f"connected as {my_peer_id}")

            # Register as listener to receive peerStatusChanged events
            self._ws.send(
                json.dumps(
                    {
                        "type": "setPeerStatus",
                        "roles": ["listener"],
                        "meta": {"name": "signalling-listener"},
                    }
                )
            )

            # Get the initial producer list
            self._ws.send(json.dumps({"type": "list"}))

            while not self._stop_event.is_set():
                try:
                    raw = self._ws.recv()
                except Exception:
                    if not self._stop_event.is_set():
                        self.logger.warning("WebSocket connection lost")
                    break

                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "list":
                    self._handle_list(msg)
                elif msg_type == "peerStatusChanged":
                    self._handle_peer_status_changed(msg)

        except Exception as e:
            if not self._stop_event.is_set():
                self.logger.error(f"error in listener loop: {e}")
        finally:
            if self._ws is not None:
                try:
                    self._ws.close()
                except Exception:
                    pass
                self._ws = None

    def _handle_list(self, msg: dict) -> None:
        """Process initial producer list."""
        for p in msg.get("producers", []):
            peer_id = p["id"]
            meta = p.get("meta", {})
            if peer_id not in self._producers:
                self._producers[peer_id] = meta
                self.logger.info(
                    f"SignallingListener: producer present: {peer_id} meta={meta}"
                )
                if self._on_producer_added:
                    self._on_producer_added(peer_id, meta)

    def _handle_peer_status_changed(self, msg: dict) -> None:
        """Process peerStatusChanged: detect producer add / remove."""
        peer_id = msg.get("peerId", "")
        roles = msg.get("roles", [])
        meta = msg.get("meta", {})

        is_producer = "producer" in roles

        if is_producer and peer_id not in self._producers:
            # New producer
            self._producers[peer_id] = meta
            self.logger.info(
                f"SignallingListener: producer added: {peer_id} meta={meta}"
            )
            if self._on_producer_added:
                self._on_producer_added(peer_id, meta)

        elif not is_producer and peer_id in self._producers:
            # Producer removed (roles became empty or no longer "producer")
            old_meta = self._producers.pop(peer_id)
            self.logger.info(
                f"SignallingListener: producer removed: {peer_id} meta={old_meta}"
            )
            if self._on_producer_removed:
                self._on_producer_removed(peer_id)


def main() -> None:
    """Get and print the gstreamer producer list."""
    parser = argparse.ArgumentParser(description="Get gstreamer producer list")
    parser.add_argument("--signalling-host", default="127.0.0.1")
    parser.add_argument("--signalling-port", default=8443, type=int)
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    producers = get_producer_list(args.signalling_host, args.signalling_port)

    if producers:
        print("List received, producers:")
        for producer_id, producer_meta in producers.items():
            print(f"  - {producer_id}: {producer_meta}")
    else:
        print("List received, no producers.")


if __name__ == "__main__":
    main()
