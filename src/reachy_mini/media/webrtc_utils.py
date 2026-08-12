"""webrtc utils functions."""

import argparse
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests
from websockets.sync.client import connect

logger = logging.getLogger(__name__)


# --- TURN relay support ----------------------------------------------
#
# By default webrtcsink only gathers host + srflx candidates, so a remote
# consumer behind a restrictive NAT (e.g. a cloud backend that can't
# UDP-hole-punch) has no way to reach the robot. Short-lived Cloudflare
# TURN credentials from HF's hosted proxy — authenticated with the
# daemon's own HF token, the same one the central relay uses — let the
# robot offer a ``relay`` candidate as well.
#
# Only the robot has to *offer* a relay: a consumer reaches it with plain
# STUN. So there is no central-server change and no consumer-side
# credential to manage (which matters because aiortc's STUN client works
# while its TURN client does not).
TURN_CREDENTIALS_URL = os.getenv(
    "REACHY_TURN_URL", "https://turn.fastrtc.org/credentials"
)
TURN_TTL_SECONDS = int(os.getenv("REACHY_TURN_TTL", "600"))

# Credentials are short-lived; re-fetch well before they expire.
_TURN_REFRESH_RATIO = 0.5
# A daemon that boots without a network shouldn't wait out a full period.
_TURN_RETRY_AFTER_FAILURE_S = 30.0
_TURN_HTTP_TIMEOUT_S = 10.0


def ice_servers_to_turn_uris(servers: List[Dict[str, Any]]) -> List[str]:
    """Convert ICE-server dicts to webrtcbin ``turn(s)://user:pass@host:port`` URIs.

    ``stun:`` entries and entries without credentials are skipped —
    webrtcbin takes a STUN server through its own ``stun-server`` property,
    and ``add-turn-server`` rejects a URI with no userinfo.

    Args:
        servers: ``{"urls", "username"?, "credential"?}`` dicts as returned
            by the TURN credentials proxy. ``urls`` may be a single string
            or a list of strings.

    Returns:
        One URI per TURN url, with credentials percent-encoded so a
        password containing ``:``, ``@`` or ``/`` can't corrupt the URI.

    """
    uris: List[str] = []
    for server in servers:
        urls = server.get("urls")
        if not urls:
            continue
        if isinstance(urls, str):
            urls = [urls]
        user = server.get("username")
        cred = server.get("credential")
        if user is None or cred is None:
            continue
        auth = f"{quote(str(user), safe='')}:{quote(str(cred), safe='')}"
        for url in urls:
            scheme, _, rest = url.partition(":")
            scheme = scheme.lower()
            if scheme in ("turn", "turns") and rest:
                uris.append(f"{scheme}://{auth}@{rest}")
    return uris


def _turn_hosts(uris: List[str]) -> List[str]:
    """Strip userinfo from TURN URIs so they can be logged safely."""
    return [u.split("@", 1)[-1] for u in uris]


class TurnCredentials:
    """Cloudflare TURN credentials kept fresh by a background thread.

    :meth:`turn_uris` never blocks and never raises: it returns whatever
    the refresher last fetched successfully. That is the whole point of
    the class — its only caller runs inside GStreamer's ``consumer-added``
    signal, where the SDP offer for that consumer cannot be generated
    until the handler returns. Fetching there would delay *every* client,
    including LAN ones that will never use a relay, by however long the
    TURN proxy takes to answer.

    Credentials expire, so the thread re-fetches at half their TTL, and
    sooner than that after a failure.
    """

    def __init__(
        self,
        url: str = TURN_CREDENTIALS_URL,
        ttl: int = TURN_TTL_SECONDS,
    ) -> None:
        """Build a refresher. Nothing is fetched until :meth:`start`.

        Args:
            url: TURN credentials proxy endpoint.
            ttl: lifetime in seconds to request for the credentials; the
                refresh period is half this.

        """
        self._url = url
        self._ttl = ttl
        # Written only by the refresher thread, read by the GStreamer
        # thread. Rebound as a whole list, never mutated in place, so a
        # reader sees either the previous set or the new one.
        self._uris: List[str] = []
        self._warned_no_token = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the refresher thread. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._refresh_loop, name="turn-credentials", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Ask the refresher thread to exit. Does not join."""
        self._stop.set()

    def turn_uris(self) -> List[str]:
        """Return the TURN URIs last fetched, or ``[]`` if none ever were."""
        return self._uris

    def _refresh_loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self._refresh_once())

    def _refresh_once(self) -> float:
        """Refresh the cache; return how long to wait before trying again.

        Only a transient failure earns the short retry. A robot that is
        simply not logged in has nothing to retry *for*, and this runs for
        the daemon's whole life — backing off fast there would put a
        warning in the log every 30 seconds forever.
        """
        period = self._ttl * _TURN_REFRESH_RATIO
        try:
            from huggingface_hub import get_token

            token = get_token()
            if not token:
                # Steady state on a robot nobody has logged in; say so once.
                if not self._warned_no_token:
                    self._warned_no_token = True
                    logger.info(
                        "No HF token; not offering TURN relay candidates. "
                        "Log in to reach this robot from outside its network."
                    )
                return period
            resp = requests.get(
                self._url,
                headers={"Authorization": f"Bearer {token}"},
                params={"ttl": self._ttl},
                timeout=_TURN_HTTP_TIMEOUT_S,
            )
            resp.raise_for_status()
            uris = ice_servers_to_turn_uris(resp.json().get("iceServers") or [])
        except Exception as e:  # noqa: BLE001 - a relay is best-effort
            logger.warning("Failed to fetch TURN credentials: %r", e)
            return _TURN_RETRY_AFTER_FAILURE_S
        self._warned_no_token = False
        if not uris:
            # The proxy answered but offered no relay: not worth hammering.
            logger.info("TURN proxy returned no relay servers")
            return period
        self._uris = uris
        logger.info("Refreshed %d TURN server(s): %s", len(uris), _turn_hosts(uris))
        return period


def get_producer_list(host: str, port: int) -> Dict[str, Dict[str, str]]:
    """Get the list of gstreamer producers from the signalling server.

    Args:
        host (str): The hostname or IP address of the signalling server.
        port (int): The port number of the signalling server.

    Returns:
        Dict[str, Dict[str, str]]: A dictionary mapping producer IDs to their metadata dictionaries.

    """
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

    Args:
        host: Host address of the signalling server.
        port: Port number of the signalling server.
        name: Producer name to search for.

    Returns:
        Peer ID of the first matching producer.

    Raises:
        KeyError: If no producer with the specified name is found.

    """
    producers = get_producer_list(host=host, port=port)

    for producer_id, producer_meta in producers.items():
        if producer_meta["name"] == name:
            return producer_id

    raise KeyError(f"Producer {name} not found.")


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
