"""WiFi Configuration Routers."""

import base64
import logging
import os
import time
from enum import Enum
from threading import Lock, Thread

import nmcli
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi import APIRouter, HTTPException
from nmcli._exception import NotExistException
from pydantic import BaseModel

from reachy_mini.utils.hardware_id import get_pin

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
                logger.exception(f"Failed to connect to WiFi network '{ssid}'")
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


# =======================
# Sealed WiFi provisioning over BLE
# =======================
# The mobile app provisions WiFi over Bluetooth without ever joining the
# robot hotspot. The BLE service (a separate systemd unit, system Python)
# relays opaque bytes to these routes; ALL crypto happens here, where
# `cryptography` is available in the daemon venv.
#
# The WiFi password is never transmitted in cleartext over BLE (a hard
# requirement for App Store review). Wire scheme `x25519-hkdf-sha256-aesgcm`:
#
#   1. Phone fetches the robot's ephemeral X25519 public key (`/wifi/prov_key`).
#   2. Phone generates its own ephemeral X25519 keypair, does ECDH, and
#      derives an AES-256-GCM key via HKDF-SHA256 with:
#         salt = device PIN (the 5-char serial suffix the user already types)
#         info = b"reachy-mini-wifi-psk-v1"
#   3. Phone seals the PSK: AES-GCM(nonce, psk, aad=ssid) and sends
#      {ssid, kid, epk, nonce, ct} to `/wifi/connect_sealed`.
#   4. Daemon repeats the ECDH+HKDF (it computes the same PIN locally via
#      get_pin()) and decrypts.
#
# AAD=ssid binds the sealed PSK to its target network (no replay onto a
# different SSID).
#
# Security scope (see BLE_WIFI_PROVISIONING.md "Security properties"):
# this protects the PSK against a PASSIVE eavesdropper — without an ECDH
# private key there is no shared secret to feed HKDF, so the password stays
# sealed. It does NOT defend against an ACTIVE man-in-the-middle. BLE pairing
# is Just-Works (NoInputNoOutput hardware → no MITM-protected pairing), and
# the phone does not verify it's talking to the real robot, so a MITM can
# answer KEYEX with its own pubkey, learn the ECDH secret, capture the sealed
# blob, and brute-force the 5-char PIN OFFLINE to open it. The wrong-PIN
# throttle only gates ONLINE guesses and does not help here. Closing this gap
# requires a PAKE (CPace/SPAKE2) that feeds the PIN into the key agreement —
# tracked as follow-up work.

# Rotate the provisioning keypair periodically; an ephemeral key has no
# long-term value, and a provisioning flow completes in seconds. A `kid`
# mismatch (e.g. rotation mid-flow) returns 400 so the phone re-fetches.
_PROV_KEY_TTL_S = 600.0
_prov_key_lock = Lock()
_prov_priv: X25519PrivateKey | None = None
_prov_kid: str | None = None
_prov_created_at: float = 0.0


class SealedConnect(BaseModel):
    """Sealed WiFi-connect payload (base64 fields, see scheme above)."""

    ssid: str
    kid: str
    epk: str  # phone ephemeral X25519 public key, raw 32B
    nonce: str  # AES-GCM nonce, 12B
    ct: str  # AES-GCM ciphertext with appended 16B tag; AAD = ssid


class _SealError(Exception):
    """Raised when a sealed payload cannot be opened (bad enc / wrong PIN)."""


def _current_provisioning_pubkey() -> tuple[str, str]:
    """Return (base64 public key, kid), rotating the keypair past its TTL."""
    global _prov_priv, _prov_kid, _prov_created_at
    with _prov_key_lock:
        now = time.monotonic()
        # Include `_prov_kid is None` so that after this block mypy can narrow
        # BOTH globals to non-None (they are always set together).
        if (
            _prov_priv is None
            or _prov_kid is None
            or (now - _prov_created_at) > _PROV_KEY_TTL_S
        ):
            _prov_priv = X25519PrivateKey.generate()
            _prov_kid = base64.urlsafe_b64encode(os.urandom(6)).decode().rstrip("=")
            _prov_created_at = now
        pub = _prov_priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        return base64.b64encode(pub).decode(), _prov_kid


def _open_sealed_psk(payload: SealedConnect) -> str:
    """Decrypt a sealed PSK. Raises _SealError on any failure."""
    with _prov_key_lock:
        if _prov_priv is None or payload.kid != _prov_kid:
            raise _SealError("unknown or expired kid")
        priv = _prov_priv
    try:
        epk = X25519PublicKey.from_public_bytes(base64.b64decode(payload.epk))
        nonce = base64.b64decode(payload.nonce)
        ct = base64.b64decode(payload.ct)
    except Exception as e:
        raise _SealError(f"bad encoding: {e}")
    shared = priv.exchange(epk)
    key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=get_pin().encode("utf-8"),
        info=b"reachy-mini-wifi-psk-v1",
    ).derive(shared)
    try:
        psk = AESGCM(key).decrypt(nonce, ct, payload.ssid.encode("utf-8"))
    except Exception:
        # Wrong PIN, tampered ciphertext, or MITM without the PIN.
        raise _SealError("authentication failed")
    return psk.decode("utf-8")


@router.get("/prov_key")
def get_provisioning_key() -> dict[str, str]:
    """Return the robot's ephemeral X25519 public key for sealed provisioning.

    Public on purpose: a bare public key is useless to an attacker who does
    not also know the device PIN (which is mixed into the key derivation).
    """
    pk_b64, kid = _current_provisioning_pubkey()
    return {"kid": kid, "pk": pk_b64, "alg": "x25519-hkdf-sha256-aesgcm"}


@router.post("/connect_sealed")
def connect_to_wifi_network_sealed(payload: SealedConnect) -> None:
    """Connect to a WiFi network using an encrypted password.

    The plaintext password never crosses BLE or the local HTTP hop — it is
    AES-GCM-sealed by the phone and opened here. See the scheme comment above.
    """
    logger.warning(f"Sealed connect request for WiFi network '{payload.ssid}'.")

    if busy_lock.locked():
        raise HTTPException(status_code=409, detail="Another operation is in progress.")

    try:
        password = _open_sealed_psk(payload)
    except _SealError as e:
        # 400 → BLE layer maps to "wrong PIN / bad credentials".
        raise HTTPException(status_code=400, detail=f"decrypt_failed: {e}")

    ssid = payload.ssid

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


# A user-triggered connect arrives while the robot is serving its own
# AP/hotspot, so NetworkManager's scan cache is stale or empty for nearby APs.
# nmcli's wifi_connect needs the target SSID in the current scan list; when it
# isn't there yet nmcli exits 10 -> NotExistException ("No network with SSID
# found"). That race is the one a fresh rescan fixes, so it's the ONLY failure
# we retry. Terminal failures are NOT retried: a wrong password surfaces as
# ConnectionActivateFailedException (nmcli exit 4), which nmcli reports without
# the underlying secrets detail and which no amount of rescanning can fix —
# retrying it would just hold busy_lock and delay the hotspot fallback by
# ~MAX_RETRIES x (scan settle + connect + delay) seconds on a hopeless connect.
WIFI_CONNECT_MAX_RETRIES = 3
WIFI_CONNECT_RETRY_DELAY = 3  # seconds between attempts
WIFI_CONNECT_SCAN_SETTLE = 2  # seconds to let the rescan populate before connecting


def _connect_station_with_rescan(ssid: str, password: str) -> None:
    """Join an AP as a station, rescanning before each attempt.

    Only the transient SSID-not-found race (see the WIFI_CONNECT_* constants
    above) is retried; every other error propagates immediately so the caller
    falls back to hotspot without delay. Re-raises the last NotExistException
    if the SSID is still missing after every rescan.
    """
    last_err: NotExistException | None = None
    for attempt in range(1, WIFI_CONNECT_MAX_RETRIES + 1):
        # Refresh the scan list after AP mode; best-effort (a rescan can
        # fail if one is already in flight, which is harmless here).
        try:
            nmcli.device.wifi_rescan()
            time.sleep(WIFI_CONNECT_SCAN_SETTLE)
        except Exception as e:
            logger.debug(f"wifi_rescan before connect failed (continuing): {e}")
        try:
            nmcli.device.wifi_connect(ssid=ssid, password=password)
            return
        except NotExistException as e:
            # SSID not in the scan list yet — the race a rescan fixes. Retry.
            last_err = e
            logger.warning(
                f"wifi_connect attempt {attempt}/{WIFI_CONNECT_MAX_RETRIES} "
                f"for '{ssid}': SSID not found yet ({e})."
            )
            if attempt < WIFI_CONNECT_MAX_RETRIES:
                time.sleep(WIFI_CONNECT_RETRY_DELAY)
    assert last_err is not None  # loop body ran at least once
    raise last_err


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
            _connect_station_with_rescan(ssid=ssid, password=password)
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
