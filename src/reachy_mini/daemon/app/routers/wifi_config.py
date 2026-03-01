"""WiFi Configuration Routers."""

import logging
import subprocess
from enum import Enum
from threading import Lock, Thread

import nmcli
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

HOTSPOT_SSID = "reachy-mini-ap"
HOTSPOT_PASSWORD = "reachy-mini"
NMCLI_COMMAND_TIMEOUT = 10  # Timeout in seconds for nmcli/iw commands


router = APIRouter(
    prefix="/wifi",
)

busy_lock = Lock()
busy_lock2 = Lock()
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
    ip_address: str | None


class SecondaryWifiStatus(BaseModel):
    """Secondary WiFi adapter (wlan1) status."""

    exists: bool
    connected: bool
    ssid: str | None
    ip_address: str | None
    known_networks: list[str]
    busy: bool


def _get_iface_ip(iface: str) -> str | None:
    """Get IP address of a network interface."""
    try:
        result = subprocess.run(
            ["ip", "-4", "addr", "show", iface],
            capture_output=True, text=True,
            timeout=NMCLI_COMMAND_TIMEOUT,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("inet "):
                return line.split()[1].split("/")[0]
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Failed to get IP for {iface}: {e}")
    return None


def get_current_wifi_mode() -> WifiMode:
    """Get the current WiFi mode."""
    if busy_lock.locked():
        return WifiMode.BUSY

    conn = get_wifi_connections()
    if check_if_connection_active("Hotspot"):
        return WifiMode.HOTSPOT
    elif any(c.device == "wlan0" for c in conn):
        return WifiMode.WLAN
    else:
        return WifiMode.DISCONNECTED


@router.get("/status")
def get_wifi_status() -> WifiStatus:
    """Get the current WiFi status."""
    mode = get_current_wifi_mode()

    connections = get_wifi_connections()
    # Filter to wlan0 connections only (exclude wlan1 secondary adapter)
    known_networks = [c.name for c in connections if c.name != "Hotspot" and not c.name.endswith("-wlan1")]

    connected_network = next((c.name for c in connections if c.device == "wlan0"), None)
    ip_address = _get_iface_ip("wlan0") if connected_network else None

    return WifiStatus(
        mode=mode,
        known_networks=known_networks,
        connected_network=connected_network,
        ip_address=ip_address,
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


# --- Secondary WiFi (wlan1) endpoints ---


def _wlan1_exists() -> bool:
    """Check if wlan1 interface exists."""
    try:
        result = subprocess.run(
            ["ip", "link", "show", "wlan1"],
            capture_output=True, text=True,
            timeout=NMCLI_COMMAND_TIMEOUT,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Failed to check wlan1 existence: {e}")
        return False


def _get_wlan1_ip() -> str | None:
    """Get IP address of wlan1."""
    return _get_iface_ip("wlan1")


def _get_wlan1_active_connection() -> str | None:
    """Get the active NM connection name on wlan1."""
    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "NAME,DEVICE", "con", "show", "--active"],
            capture_output=True, text=True,
            timeout=NMCLI_COMMAND_TIMEOUT,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            parts = line.split(":")
            if len(parts) >= 2 and parts[-1] == "wlan1":
                return parts[0]
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Failed to get wlan1 active connection: {e}")
    return None


def _get_wlan1_ssid() -> str | None:
    """Get the SSID that wlan1 is connected to."""
    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "DEVICE,ACTIVE,SSID", "dev", "wifi"],
            capture_output=True, text=True,
            timeout=NMCLI_COMMAND_TIMEOUT,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            parts = line.split(":")
            if len(parts) >= 3 and parts[0] == "wlan1" and parts[1] == "yes":
                return parts[2]
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Failed to get wlan1 SSID: {e}")
    return None


def _get_wlan1_known_networks() -> list[str]:
    """Get saved NM connection profiles bound to wlan1."""
    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "NAME,TYPE,DEVICE", "con", "show"],
            capture_output=True, text=True,
            timeout=NMCLI_COMMAND_TIMEOUT,
        )
        if result.returncode != 0:
            return []
        networks = []
        for line in result.stdout.splitlines():
            parts = line.split(":")
            if len(parts) >= 3 and parts[1] == "802-11-wireless" and parts[2] == "wlan1":
                networks.append(parts[0])
        return networks
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Failed to get wlan1 known networks: {e}")
        return []


def _ensure_wlan1() -> bool:
    """Create wlan1 virtual interface if it doesn't exist. Returns True if exists/created."""
    if _wlan1_exists():
        return True
    logger.info("Creating wlan1 virtual interface...")
    try:
        subprocess.run(
            ["sudo", "iw", "dev", "wlan0", "interface", "add", "wlan1", "type", "managed"],
            capture_output=True, text=True,
            timeout=NMCLI_COMMAND_TIMEOUT,
            check=True,
        )
        subprocess.run(
            ["sudo", "ip", "link", "set", "wlan1", "up"],
            capture_output=True, text=True,
            timeout=NMCLI_COMMAND_TIMEOUT,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create wlan1: {e.stderr}")
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Failed to create wlan1: {e}")
        return False


@router.get("/status2")
def get_secondary_wifi_status() -> SecondaryWifiStatus:
    """Get secondary WiFi adapter (wlan1) status."""
    if not _wlan1_exists():
        return SecondaryWifiStatus(
            exists=False, connected=False, ssid=None, ip_address=None,
            known_networks=[], busy=busy_lock2.locked(),
        )

    ssid = _get_wlan1_ssid()
    ip_addr = _get_wlan1_ip()
    connected = ssid is not None and ip_addr is not None
    known = _get_wlan1_known_networks()

    return SecondaryWifiStatus(
        exists=True,
        connected=connected,
        ssid=ssid,
        ip_address=ip_addr,
        known_networks=known,
        busy=busy_lock2.locked(),
    )


@router.post("/scan2")
def scan_secondary_wifi() -> list[str]:
    """Scan for networks visible from wlan1."""
    if not _wlan1_exists():
        raise HTTPException(status_code=404, detail="wlan1 interface does not exist.")

    try:
        subprocess.run(
            ["nmcli", "dev", "wifi", "rescan", "ifname", "wlan1"],
            capture_output=True, text=True,
            timeout=NMCLI_COMMAND_TIMEOUT,
        )
        result = subprocess.run(
            ["nmcli", "-t", "-f", "SSID,SIGNAL", "dev", "wifi", "list", "ifname", "wlan1"],
            capture_output=True, text=True,
            timeout=NMCLI_COMMAND_TIMEOUT,
        )
        if result.returncode != 0:
            return []

        seen = set()
        ssids = []
        for line in result.stdout.splitlines():
            parts = line.rsplit(":", 1)
            ssid = parts[0].strip()
            if ssid and ssid not in seen:
                seen.add(ssid)
                ssids.append(ssid)
        return ssids
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Failed to scan wlan1 networks: {e}")
        return []


@router.post("/connect2")
def connect_secondary_wifi(ssid: str, password: str) -> None:
    """Connect wlan1 to a network."""
    if busy_lock2.locked():
        raise HTTPException(status_code=409, detail="Another operation is in progress on wlan1.")

    def connect() -> None:
        with busy_lock2:
            try:
                if not _ensure_wlan1():
                    logger.error("Cannot create wlan1 interface")
                    return

                con_name = f"{ssid}-wlan1"

                # Check if connection profile already exists
                check = subprocess.run(
                    ["nmcli", "-t", "-f", "NAME", "con", "show"],
                    capture_output=True, text=True,
                    timeout=NMCLI_COMMAND_TIMEOUT,
                )
                profile_exists = con_name in check.stdout.splitlines()

                if profile_exists:
                    # Update password and bring up
                    subprocess.run(
                        ["nmcli", "con", "modify", con_name,
                         "wifi-sec.key-mgmt", "wpa-psk",
                         "wifi-sec.psk", password],
                        capture_output=True, text=True,
                        timeout=NMCLI_COMMAND_TIMEOUT,
                    )
                    result = subprocess.run(
                        ["nmcli", "con", "up", con_name],
                        capture_output=True, text=True,
                        timeout=NMCLI_COMMAND_TIMEOUT,
                    )
                else:
                    # Create new connection profile
                    result = subprocess.run(
                        ["nmcli", "con", "add",
                         "type", "wifi",
                         "ifname", "wlan1",
                         "con-name", con_name,
                         "ssid", ssid,
                         "wifi-sec.key-mgmt", "wpa-psk",
                         "wifi-sec.psk", password],
                        capture_output=True, text=True,
                        timeout=NMCLI_COMMAND_TIMEOUT,
                    )
                    if result.returncode != 0:
                        logger.error(f"Failed to create connection for {ssid}: {result.stderr}")
                        return

                    result = subprocess.run(
                        ["nmcli", "con", "up", con_name],
                        capture_output=True, text=True,
                        timeout=NMCLI_COMMAND_TIMEOUT,
                    )

                if result.returncode != 0:
                    logger.error(f"Failed to connect wlan1 to {ssid}: {result.stderr}")
                else:
                    logger.info(f"wlan1 connected to {ssid}")

            except Exception as e:
                logger.error(f"Error connecting wlan1 to {ssid}: {e}")

    Thread(target=connect).start()


@router.post("/disconnect2")
def disconnect_secondary_wifi() -> None:
    """Disconnect wlan1."""
    if busy_lock2.locked():
        raise HTTPException(status_code=409, detail="Another operation is in progress on wlan1.")

    con_name = _get_wlan1_active_connection()
    if not con_name:
        raise HTTPException(status_code=400, detail="wlan1 has no active connection.")

    def disconnect() -> None:
        with busy_lock2:
            try:
                subprocess.run(
                    ["nmcli", "con", "down", con_name],
                    capture_output=True, text=True,
                    timeout=NMCLI_COMMAND_TIMEOUT,
                    check=True,
                )
                logger.info(f"wlan1 disconnected from {con_name}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to disconnect wlan1: {e.stderr}")
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                logger.error(f"Error disconnecting wlan1: {e}")

    Thread(target=disconnect).start()


@router.post("/create_interface")
def create_secondary_interface() -> dict[str, str]:
    """Create wlan1 virtual interface if missing."""
    if _wlan1_exists():
        return {"status": "already_exists"}
    if _ensure_wlan1():
        return {"status": "created"}
    raise HTTPException(status_code=500, detail="Failed to create wlan1 interface.")


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


WIFI_INIT_MAX_RETRIES = 5
WIFI_INIT_RETRY_DELAY = 3  # seconds
WIFI_INIT_TIMEOUT = 30  # seconds


def ensure_wifi_on_startup() -> None:
    """Ensure WiFi is configured on daemon startup.

    Retries if NetworkManager or the WiFi interface isn't ready yet.
    On final failure the daemon keeps running so the robot stays
    reachable via Bluetooth for recovery.
    """
    import time

    for attempt in range(1, WIFI_INIT_MAX_RETRIES + 1):
        try:
            # Make sure wlan0 is up and running
            scan_available_wifi()

            # If no WiFi connection is active, set up the default hotspot
            if get_current_wifi_mode() == WifiMode.DISCONNECTED:
                logger.info("No WiFi connection active. Setting up hotspot...")
                setup_wifi_connection(
                    name="Hotspot",
                    ssid=HOTSPOT_SSID,
                    password=HOTSPOT_PASSWORD,
                    is_hotspot=True,
                )
            return
        except Exception as e:
            logger.warning(
                f"WiFi init attempt {attempt}/{WIFI_INIT_MAX_RETRIES} failed: {e}"
            )
            if attempt < WIFI_INIT_MAX_RETRIES:
                time.sleep(WIFI_INIT_RETRY_DELAY)

    logger.error(
        f"WiFi initialization failed after {WIFI_INIT_MAX_RETRIES} attempts. "
        "Daemon will start without WiFi configured."
    )


_wifi_init_thread = Thread(target=ensure_wifi_on_startup, daemon=True)
_wifi_init_thread.start()
_wifi_init_thread.join(timeout=WIFI_INIT_TIMEOUT)
if _wifi_init_thread.is_alive():
    logger.error(
        f"WiFi initialization timed out after {WIFI_INIT_TIMEOUT}s. "
        "Daemon will start without WiFi configured."
    )
