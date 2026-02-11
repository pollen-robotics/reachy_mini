"""webrtc utils functions."""

import argparse
import json
import logging
from typing import Dict

from websockets.sync.client import connect


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
