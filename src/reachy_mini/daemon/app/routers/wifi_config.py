"""WiFi Configuration Routers."""

import logging
from enum import Enum
from threading import Lock, Thread

import nmcli
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

HOTSPOT_SSID = "reachy-mini-ap"
HOTSPOT_PASSWORD = "reachy-mini"


router = APIRouter(
    prefix="/wifi",
)

busy_lock = Lock()
error: Exception | None = None
logger = logging.getLogger(__name__)


class WifiMode(Enum):
    """WiFi possible modes."""

    HOTSPOT = "hotspot"
    WLAN = "wlan"
    DISCONNECTED = "disconnected"
    BUSY = "busy"


class WifiStatus(BaseModel):
    """WiFi status model."""

    mode: WifiMode
    known_networks: list[str]
    connected_network: str | None


def get_current_wifi_mode() -> WifiMode:
    """Get the current WiFi mode."""
    if busy_lock.locked():
        return WifiMode.BUSY

    conn = get_wifi_connections()
    if check_if_connection_active("Hotspot"):
        return WifiMode.HOTSPOT
    elif any(c.device != "--" for c in conn):
        return WifiMode.WLAN
    else:
        return WifiMode.DISCONNECTED


@router.get("/status")
def get_wifi_status() -> WifiStatus:
    """Get the current WiFi status."""
    mode = get_current_wifi_mode()

    connections = get_wifi_connections()
    known_networks = [c.name for c in connections if c.name != "Hotspot"]

    connected_network = next((c.name for c in connections if c.device != "--"), None)

    return WifiStatus(
        mode=mode,
        known_networks=known_networks,
        connected_network=connected_network,
    )


@router.get("/error")
def get_last_wifi_error() -> dict[str, str | None]:
    """Get the last WiFi error."""
    global error
    if error is None:
        return {"error": None}
    return {"error": str(error)}


@router.post("/reset_error")
def reset_last_wifi_error() -> dict[str, str]:
    """Reset the last WiFi error."""
    global error
    error = None
    return {"status": "ok"}


@router.post("/setup_hotspot")
def setup_hotspot(
    ssid: str = HOTSPOT_SSID,
    password: str = HOTSPOT_PASSWORD,
) -> None:
    """Set up a WiFi hotspot. It will create a new hotspot using nmcli if one does not already exist."""
    if busy_lock.locked():
        raise HTTPException(status_code=409, detail="Another operation is in progress.")

    def hotspot() -> None:
        with busy_lock:
            setup_wifi_connection(
                name="Hotspot", ssid=ssid, password=password, is_hotspot=True
            )

    Thread(target=hotspot).start()
    # TODO: wait for it to be really started


@router.post("/connect")
def connect_to_wifi_network(
    ssid: str,
    password: str,
) -> None:
    """Connect to a WiFi network. It will create a new connection using nmcli if the specified SSID is not already configured."""
    logger.warning(f"Request to connect to WiFi network '{ssid}' received.")

    if busy_lock.locked():
        raise HTTPException(status_code=409, detail="Another operation is in progress.")

    def connect() -> None:
        global error
        with busy_lock:
            try:
                error = None
                setup_wifi_connection(name=ssid, ssid=ssid, password=password)
            except Exception as e:
                error = e
                logger.error(f"Failed to connect to WiFi network '{ssid}': {e}")
                logger.info("Reverting to hotspot...")
                remove_connection(name=ssid)
                setup_wifi_connection(
                    name="Hotspot",
                    ssid=HOTSPOT_SSID,
                    password=HOTSPOT_PASSWORD,
                    is_hotspot=True,
                )

    Thread(target=connect).start()
    # TODO: wait for it to be really connected


@router.post("/scan_and_list")
def scan_wifi() -> list[str]:
    """Scan for available WiFi networks ordered by signal power."""
    wifi = scan_available_wifi()

    seen = set()
    ssids = [x.ssid for x in wifi if x.ssid not in seen and not seen.add(x.ssid)]  # type: ignore

    return ssids


@router.post("/forget")
def forget_wifi_network(ssid: str) -> None:
    """Forget a saved WiFi network. Falls back to Hotspot if forgetting the active network."""
    if ssid == "Hotspot":
        raise HTTPException(status_code=400, detail="Cannot forget Hotspot connection.")

    if not check_if_connection_exists(ssid):
        raise HTTPException(
            status_code=404, detail=f"Network '{ssid}' not found in saved networks."
        )

    if busy_lock.locked():
        raise HTTPException(status_code=409, detail="Another operation is in progress.")

    def forget() -> None:
        global error
        with busy_lock:
            try:
                error = None
                was_active = check_if_connection_active(ssid)
                logger.info(f"Forgetting WiFi network '{ssid}'...")
                remove_connection(ssid)

                if was_active:
                    logger.info("Was connected, falling back to hotspot...")
                    setup_wifi_connection(
                        name="Hotspot",
                        ssid=HOTSPOT_SSID,
                        password=HOTSPOT_PASSWORD,
                        is_hotspot=True,
                    )
            except Exception as e:
                error = e
                logger.error(f"Failed to forget network '{ssid}': {e}")

    Thread(target=forget).start()


@router.post("/forget_all")
def forget_all_wifi_networks() -> None:
    """Forget all saved WiFi networks (except Hotspot). Falls back to Hotspot."""
    if busy_lock.locked():
        raise HTTPException(status_code=409, detail="Another operation is in progress.")

    def forget_all() -> None:
        global error
        with busy_lock:
            try:
                error = None
                connections = get_wifi_connections()
                forgotten = []

                for conn in connections:
                    if conn.name != "Hotspot":
                        remove_connection(conn.name)
                        forgotten.append(conn.name)

                logger.info(f"Forgotten {len(forgotten)} networks: {forgotten}")

                # Always ensure we have connectivity after forgetting all
                if get_current_wifi_mode() == WifiMode.DISCONNECTED:
                    logger.info("No connection left, setting up hotspot...")
                    setup_wifi_connection(
                        name="Hotspot",
                        ssid=HOTSPOT_SSID,
                        password=HOTSPOT_PASSWORD,
                        is_hotspot=True,
                    )
            except Exception as e:
                error = e
                logger.error(f"Failed to forget networks: {e}")

    Thread(target=forget_all).start()


# NMCLI WRAPPERS
def scan_available_wifi() -> list[nmcli.data.device.DeviceWifi]:
    """Scan for available WiFi networks."""
    nmcli.device.wifi_rescan()
    devices: list[nmcli.data.device.DeviceWifi] = nmcli.device.wifi()
    return devices


def get_wifi_connections() -> list[nmcli.data.connection.Connection]:
    """Get the list of WiFi connection."""
    return [conn for conn in nmcli.connection() if conn.conn_type == "wifi"]


def check_if_connection_exists(name: str) -> bool:
    """Check if a WiFi connection with the given SSID already exists."""
    return any(c.name == name for c in get_wifi_connections())


def check_if_connection_active(name: str) -> bool:
    """Check if a WiFi connection with the given SSID is currently active."""
    return any(c.name == name and c.device != "--" for c in get_wifi_connections())


def setup_wifi_connection(
    name: str, ssid: str, password: str, is_hotspot: bool = False
) -> None:
    """Set up a WiFi connection using nmcli."""
    logger.info(f"Setting up WiFi connection (ssid='{ssid}')...")

    if not check_if_connection_exists(name):
        logger.info("WiFi configuration does not exist. Creating...")
        if is_hotspot:
            nmcli.device.wifi_hotspot(ssid=ssid, password=password)
        else:
            nmcli.device.wifi_connect(ssid=ssid, password=password)
        return

    logger.info("WiFi configuration already exists.")
    if not check_if_connection_active(name):
        logger.info("WiFi is not active. Activating...")
        nmcli.connection.up(name)
        return

    logger.info(f"Connection {name} is already active.")


def remove_connection(name: str) -> None:
    """Remove a WiFi connection using nmcli."""
    if check_if_connection_exists(name):
        logger.info(f"Removing WiFi connection '{name}'...")
        nmcli.connection.delete(name)


# Setup WiFi connection on startup

# This make sure the wlan0 is up and running
scan_available_wifi()

# On startup, if no WiFi connection is active, set up the default hotspot
if get_current_wifi_mode() == WifiMode.DISCONNECTED:
    logger.info("No WiFi connection active. Setting up hotspot...")

    setup_wifi_connection(
        name="Hotspot",
        ssid=HOTSPOT_SSID,
        password=HOTSPOT_PASSWORD,
        is_hotspot=True,
    )
