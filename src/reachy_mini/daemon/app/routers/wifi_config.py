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
HOTSPOT_CONNECTION_NAME = "Hotspot"


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


def compute_wifi_mode() -> WifiMode:
    """Compute the real WiFi mode, ignoring busy state.

    Unlike ``get_current_wifi_mode``, this never returns ``BUSY`` and can be
    safely called from code that already holds ``busy_lock`` (e.g. worker
    threads inside ``with busy_lock:``).
    """
    conn = get_wifi_connections()
    if check_if_connection_active(HOTSPOT_CONNECTION_NAME):
        return WifiMode.HOTSPOT
    elif any(c.device != "--" for c in conn):
        return WifiMode.WLAN
    else:
        return WifiMode.DISCONNECTED


def get_current_wifi_mode() -> WifiMode:
    """Get the current WiFi mode."""
    if busy_lock.locked():
        return WifiMode.BUSY
    return compute_wifi_mode()


@router.get("/status")
def get_wifi_status() -> WifiStatus:
    """Get the current WiFi status."""
    mode = get_current_wifi_mode()

    connections = get_wifi_connections()
    known_networks = [c.name for c in connections if c.name != HOTSPOT_CONNECTION_NAME]

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
            # Use the default hotspot helper when the caller didn't override
            # credentials so we benefit from the auto-heal behavior.
            if ssid == HOTSPOT_SSID and password == HOTSPOT_PASSWORD:
                ensure_hotspot_active()
            else:
                setup_wifi_connection(
                    name=HOTSPOT_CONNECTION_NAME,
                    ssid=ssid,
                    password=password,
                    is_hotspot=True,
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
                ensure_hotspot_active()

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
    if ssid == HOTSPOT_CONNECTION_NAME:
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
                    ensure_hotspot_active()
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

                # Check BEFORE deletion if we were actively connected to a
                # non-Hotspot network. We cannot rely on the post-delete mode
                # because deleting the active profile can leave wlan0 in a
                # transient state where no wifi is "active" yet no error is
                # raised.
                was_connected_to_wifi = any(
                    c.name != HOTSPOT_CONNECTION_NAME and c.device != "--"
                    for c in connections
                )

                for conn in connections:
                    if conn.name != HOTSPOT_CONNECTION_NAME:
                        remove_connection(conn.name)
                        forgotten.append(conn.name)

                logger.info(f"Forgotten {len(forgotten)} networks: {forgotten}")

                # NOTE: ``get_current_wifi_mode()`` would return ``BUSY`` here
                # because we hold ``busy_lock``. We use ``compute_wifi_mode``
                # directly so the fallback actually triggers.
                if was_connected_to_wifi or compute_wifi_mode() != WifiMode.HOTSPOT:
                    logger.info("Falling back to hotspot after clearing networks...")
                    ensure_hotspot_active()
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


def _prepare_wlan0_for_hotspot() -> None:
    """Best-effort hygiene before (re)starting the hotspot on wlan0.

    Mirrors the behavior of ``bluetooth/commands/HOTSPOT.sh`` so the on-device
    recovery paths (WiFi API + BLE command) behave identically. Any failure is
    logged but never raised: we want to keep trying to bring the AP up even if
    one of these preparation steps is unavailable on the host.
    """
    for cmd in (
        ["nmcli", "device", "disconnect", "wlan0"],
        ["rfkill", "unblock", "wifi"],
    ):
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except Exception as exc:  # noqa: BLE001 - best effort
            logger.debug(f"Preparation step {cmd!r} failed: {exc}")


def ensure_hotspot_active() -> None:
    """Make sure the Hotspot AP is up, auto-healing a broken profile if needed.

    ``nmcli.connection.up("Hotspot")`` can silently fail when the stored
    profile is stale (e.g. wrong mode, missing ``ipv4.method=shared``) or when
    wlan0 is in a weird state after a connection was deleted. In that case we
    drop the profile and re-create it from scratch via ``wifi_hotspot`` so the
    robot always ends up reachable on ``10.42.0.1``.
    """
    _prepare_wlan0_for_hotspot()

    try:
        setup_wifi_connection(
            name=HOTSPOT_CONNECTION_NAME,
            ssid=HOTSPOT_SSID,
            password=HOTSPOT_PASSWORD,
            is_hotspot=True,
        )
    except Exception as exc:  # noqa: BLE001 - we want to attempt recovery
        logger.warning(
            f"Failed to activate existing Hotspot profile ({exc}). "
            "Recreating it from scratch..."
        )
        try:
            remove_connection(HOTSPOT_CONNECTION_NAME)
        except Exception as remove_exc:  # noqa: BLE001
            logger.warning(f"Failed to remove stale Hotspot profile: {remove_exc}")
        nmcli.device.wifi_hotspot(ssid=HOTSPOT_SSID, password=HOTSPOT_PASSWORD)


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

            # If no WiFi connection is active, set up the default hotspot.
            # ``compute_wifi_mode`` is used instead of ``get_current_wifi_mode``
            # for symmetry with ``forget_all`` and to avoid any surprise if a
            # future refactor schedules this on a thread that holds the lock.
            if compute_wifi_mode() == WifiMode.DISCONNECTED:
                logger.info("No WiFi connection active. Setting up hotspot...")
                ensure_hotspot_active()
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
