"""Read-only smoke tests for the Reachy Mini BLE provisioning service.

Run over the air from an external host (see conftest.py). Every test here is
safe by construction: no PIN is ever sent (so the robot-global wrong-PIN
lockout is never armed), no state on the robot is mutated, and no CMD_*
script is executed. The auth-gate tests assert that privileged commands are
REFUSED without a session — which is exactly the unauthenticated path.

Commands the deployed firmware may predate (sealed WiFi provisioning, the
Hardware ID characteristic) skip with an explanation instead of failing:
unknown commands are echoed back by design, which is how a client detects an
older robot.
"""

import base64
import json
import re
import uuid

import pytest

from .ble_link import (
    AVAILABLE_COMMANDS_UUID,
    COMMAND_CHAR_UUID,
    COMMANDS_DIR,
    FIRMWARE_REVISION_UUID,
    HARDWARE_ID_UUID,
    MANUFACTURER_NAME_UUID,
    MODEL_NUMBER_UUID,
    NETWORK_STATUS_UUID,
    REACHY_STATUS_SERVICE_UUID,
    RESPONSE_CHAR_UUID,
    SYSTEM_STATUS_UUID,
    ReachyBleLink,
)

AUTH_ERROR = "ERROR: Not connected. Please authenticate first."


def _skip_if_unsupported(response: str, command: str) -> None:
    if response.startswith("ECHO:"):
        pytest.skip(f"robot firmware predates {command} (command echoed back)")


# -- discovery / GATT layout -------------------------------------------


def test_robot_is_discoverable(link: ReachyBleLink) -> None:
    """The robot is found by a plain BLE scan, with no prior knowledge."""
    assert link.device_address is not None
    assert link.device_name is not None and "reachy" in link.device_name.lower()


def test_advertises_status_service_uuid(link: ReachyBleLink) -> None:
    """The advertisement carries the Reachy Status service UUID."""
    assert REACHY_STATUS_SERVICE_UUID in link.advertised_uuids


def test_command_service_gatt_layout(link: ReachyBleLink) -> None:
    """Command service exposes write-only command + read/notify response."""
    command_char = link.characteristic(COMMAND_CHAR_UUID)
    response_char = link.characteristic(RESPONSE_CHAR_UUID)
    assert command_char is not None and "write" in command_char.properties
    assert response_char is not None
    assert {"read", "notify"} <= set(response_char.properties)


def test_device_information_service(link: ReachyBleLink) -> None:
    """Standard DIS identifies the robot; firmware rev encodes hotspot IP."""
    assert link.read_char(MANUFACTURER_NAME_UUID) == "Pollen Robotics"
    assert link.read_char(MODEL_NUMBER_UUID) == "Reachy Mini"
    assert link.read_char(FIRMWARE_REVISION_UUID).startswith("[HOTSPOT]:")


# -- status characteristics --------------------------------------------


def test_system_status_is_online(link: ReachyBleLink) -> None:
    """System status characteristic reads as Online."""
    assert link.read_char(SYSTEM_STATUS_UUID) == "Online"


def test_network_status_format(link: ReachyBleLink) -> None:
    """Network status matches the documented MODE [iface] ip format."""
    status = link.read_char(NETWORK_STATUS_UUID)
    assert re.fullmatch(
        r"OFFLINE|ERROR|(HOTSPOT|CONNECTED)( \[\S+\] \S+( ;)?)+", status
    ), f"unexpected network status: {status!r}"


def test_available_commands_match_repo(link: ReachyBleLink) -> None:
    """The robot reports at least every CMD_* script shipped in this repo."""
    reported = {c.strip() for c in link.read_char(AVAILABLE_COMMANDS_UUID).split(",")}
    expected = {p.stem for p in COMMANDS_DIR.glob("*.sh")}
    assert expected, "no commands/*.sh found in the installed package"
    missing = expected - reported
    assert not missing, (
        f"robot does not expose {sorted(missing)} (has {sorted(reported)})"
    )


def test_hardware_id_characteristic(link: ReachyBleLink) -> None:
    """Hardware ID reads as 16 lowercase hex chars (or 'unknown')."""
    if link.characteristic(HARDWARE_ID_UUID) is None:
        pytest.skip("robot firmware predates the Hardware ID characteristic")
    hardware_id = link.read_char(HARDWARE_ID_UUID)
    assert hardware_id == "unknown" or re.fullmatch(r"[0-9a-f]{16}", hardware_id), (
        f"unexpected hardware id: {hardware_id!r}"
    )


# -- synchronous command round-trips -----------------------------------


def test_ping(link: ReachyBleLink) -> None:
    """PING returns PONG."""
    assert link.command("PING") == "PONG"


def test_echo_roundtrip(link: ReachyBleLink) -> None:
    """Unknown commands are echoed back verbatim — proves the write/read path."""
    nonce = f"smoke-{uuid.uuid4().hex[:12]}"
    assert link.command(nonce) == f"ECHO: {nonce}"


def test_journal_stream_lifecycle(link: ReachyBleLink) -> None:
    """JOURNAL_START/READ/STOP round-trip, leaving streaming stopped."""
    started = link.command("JOURNAL_START")
    assert started in ("OK: Journal streaming started", "OK: Journal already streaming")
    try:
        read = link.command("JOURNAL_READ")
        assert not read.startswith("ERROR"), f"journal read failed: {read!r}"
    finally:
        assert link.command("JOURNAL_STOP") == "OK: Journal streaming stopped"
    assert link.command("JOURNAL_READ") == "ERROR: Journal not running"


# -- WiFi provisioning, public surface ---------------------------------


def test_wifi_status_public_omits_known_networks(link: ReachyBleLink) -> None:
    """Unauthenticated WIFI_STATUS must NOT leak the saved-network list."""
    response = link.command("WIFI_STATUS")
    _skip_if_unsupported(response, "WIFI_STATUS")
    status = json.loads(response)
    assert {"mode", "connected"} <= set(status), status
    assert "known" not in status, "saved networks leaked without authentication"


def test_wifi_keyex_returns_fresh_x25519_key(link: ReachyBleLink) -> None:
    """WIFI_KEYEX is public and returns a valid provisioning pubkey."""
    response = link.command("WIFI_KEYEX")
    _skip_if_unsupported(response, "WIFI_KEYEX")
    key = json.loads(response)
    assert key.get("alg") == "x25519-hkdf-sha256-aesgcm", key
    assert key.get("kid"), key
    assert len(base64.b64decode(key["pk"])) == 32


# -- auth gates (no PIN is ever sent) ----------------------------------


def test_wifi_scan_requires_auth(link: ReachyBleLink) -> None:
    """WIFI_SCAN is refused without a PIN session."""
    response = link.command("WIFI_SCAN")
    _skip_if_unsupported(response, "WIFI_SCAN")
    assert response == AUTH_ERROR


def test_cmd_scripts_require_auth(link: ReachyBleLink) -> None:
    """CMD_* scripts are refused without a session (nothing is executed)."""
    assert link.command("CMD_RESTART_DAEMON") == AUTH_ERROR
