"""WiFi Configuration Routers."""

import logging
from enum import Enum
from threading import Lock, Thread

from fastapi import APIRouter, HTTPException

HOTSPOT_SSID = "reachy-mini-ap"
HOTSPOT_PASSWORD = "reachy-mini"


try:
    import nmcli

    router = APIRouter(
        prefix="/wifi_config",
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

    @router.get("/status")
    async def get_current_wifi_mode() -> WifiMode:
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

    @router.get("/error")
    def get_last_wifi_error():
        """Get the last WiFi error."""
        global error
        if error is None:
            return {"error": None}
        return {"error": str(error)}

    @router.post("/reset_error")
    def reset_last_wifi_error():
        """Reset the last WiFi error."""
        global error
        error = None
        return {"status": "ok"}

    @router.post("/setup_hotspot")
    def setup_hotspot(
        ssid: str = HOTSPOT_SSID,
        password: str = HOTSPOT_PASSWORD,
    ):
        """Set up a WiFi hotspot. It will create a new hotspot using nmcli if one does not already exist."""
        if busy_lock.locked():
            raise HTTPException(
                status_code=409, detail="Another operation is in progress."
            )

        def hotspot():
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
    ):
        """Connect to a WiFi network. It will create a new connection using nmcli if the specified SSID is not already configured."""
        logger.warning(f"Request to connect to WiFi network '{ssid}' received.")

        if busy_lock.locked():
            raise HTTPException(
                status_code=409, detail="Another operation is in progress."
            )

        def connect():
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

    # NMCLI WRAPPERS
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
    ):
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

    def remove_connection(name: str):
        """Remove a WiFi connection using nmcli."""
        if check_if_connection_exists(name):
            logger.info(f"Removing WiFi connection '{name}'...")
            nmcli.connection.delete(name)

    if get_current_wifi_mode() == WifiMode.DISCONNECTED:
        logger.info("No WiFi connection active. Setting up hotspot...")

        setup_wifi_connection(
            name="Hotspot",
            ssid=HOTSPOT_SSID,
            password=HOTSPOT_PASSWORD,
            is_hotspot=True,
        )
except ImportError:
    pass
