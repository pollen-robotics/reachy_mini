"""Plumbing for the over-the-air BLE test suite.

This module gives the tests two things:

1. The *protocol constants* (UUIDs), parsed from the real
   ``bluetooth_service.py`` source shipped in the same wheel, so the suite
   fails loudly (KeyError at import) if the firmware protocol drifts instead
   of silently testing stale literals.

2. ``ReachyBleLink`` — a synchronous client that scans for, connects to, and
   exchanges commands with a live Reachy Mini over the air. Scanning shells
   out to ``bluetoothctl`` (present on any BlueZ host, keeps the suite free
   of third-party dependencies); the connection itself uses a raw LE ATT
   socket (see att_client.py) because BlueZ's high-level Connect() picks the
   classic BR/EDR bearer for the robot's dual-mode advertisement and fails —
   phones and Web Bluetooth are LE-only and never hit this, and this client
   behaves like them. The robot is exercised exactly as shipped.

The suite is meant to run on an external Linux host with a BLE adapter (e.g.
a Raspberry Pi) sitting in radio range of the robot — NOT on the robot.
"""

import importlib.util
import re
import subprocess
from pathlib import Path

from .att_client import AttClient, Characteristic


def _reachy_mini_package_dir() -> Path:
    """Locate the installed reachy_mini package WITHOUT importing it.

    find_spec resolves the package location but does not execute its
    __init__ (which chains into numpy/scipy and friends) — this keeps the
    suite runnable on a minimal test host installed with ``pip install
    --no-deps reachy-mini pytest``. Falls back to the sibling directory for
    a plain source checkout.
    """
    spec = importlib.util.find_spec("reachy_mini")
    if spec is not None and spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations)))
    return Path(__file__).resolve().parents[1] / "reachy_mini"


# The BLE service and its CMD_* scripts ship in the same wheel.
_SERVICE_PATH = (
    _reachy_mini_package_dir() / "daemon/app/services/bluetooth/bluetooth_service.py"
)
COMMANDS_DIR = _SERVICE_PATH.parent / "commands"

# Protocol constants, parsed straight from the firmware source.
_UUIDS: dict[str, str] = dict(
    re.findall(r'^(\w+_UUID) = "([0-9a-f-]+)"', _SERVICE_PATH.read_text(), re.M)
)
COMMAND_CHAR_UUID = _UUIDS["COMMAND_CHAR_UUID"]
RESPONSE_CHAR_UUID = _UUIDS["RESPONSE_CHAR_UUID"]
REACHY_STATUS_SERVICE_UUID = _UUIDS["REACHY_STATUS_SERVICE_UUID"]
MANUFACTURER_NAME_UUID = _UUIDS["MANUFACTURER_NAME_UUID"]
MODEL_NUMBER_UUID = _UUIDS["MODEL_NUMBER_UUID"]
FIRMWARE_REVISION_UUID = _UUIDS["FIRMWARE_REVISION_UUID"]
NETWORK_STATUS_UUID = _UUIDS["NETWORK_STATUS_UUID"]
SYSTEM_STATUS_UUID = _UUIDS["SYSTEM_STATUS_UUID"]
AVAILABLE_COMMANDS_UUID = _UUIDS["AVAILABLE_COMMANDS_UUID"]
HARDWARE_ID_UUID = _UUIDS["HARDWARE_ID_UUID"]

# The advertised LocalName is "ReachyMini" but BlueZ may cache the GAP Device
# Name ("reachy-mini", the hostname) instead, depending on scan history. Match
# on a normalized form so both spellings pass.
DEFAULT_DEVICE_NAME = "ReachyMini"


def _normalize_name(name: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


class BleLinkError(RuntimeError):
    """Raised when the BLE link cannot be established or times out."""


class ReachyBleLink:
    """Synchronous BLE client for a Reachy Mini.

    Discovery shells out to bluetoothctl (works unprivileged); the connection
    uses a raw LE ATT socket (att_client.py) to force the LE bearer.
    """

    # Immediate ack returned by daemon-proxied commands; the real payload
    # follows as a notification on the response characteristic.
    ACK = "OK: working"

    def __init__(
        self,
        name: str = DEFAULT_DEVICE_NAME,
        address: str | None = None,
        scan_timeout: float = 15.0,
        response_timeout: float = 25.0,
    ) -> None:
        """Configure the link (no I/O happens until discover())."""
        self.name = name
        self.address = address
        self.scan_timeout = scan_timeout
        self.response_timeout = response_timeout
        self.device_address: str | None = None
        self.device_name: str | None = None
        self.advertised_uuids: set[str] = set()
        self._att: AttClient | None = None
        self._command_char: Characteristic | None = None
        self._response_char: Characteristic | None = None

    # -- discovery / connection ----------------------------------------

    @staticmethod
    def _bluetoothctl(*args: str, timeout: float) -> str:
        return subprocess.run(
            ["bluetoothctl", *args], capture_output=True, text=True, timeout=timeout
        ).stdout

    def discover(self) -> str | None:
        """Scan for the robot; returns its MAC address or None.

        A device matches if its name normalizes to the expected name. When
        ``address`` was given, only that address is considered. The UUIDs
        BlueZ learned from the device's advertisement are captured too.
        """
        self._bluetoothctl(
            "--timeout",
            str(int(self.scan_timeout)),
            "scan",
            "on",
            timeout=self.scan_timeout + 10,
        )
        wanted = _normalize_name(self.name)
        for line in self._bluetoothctl("devices", timeout=10).splitlines():
            match = re.fullmatch(r"Device ((?:[0-9A-F]{2}:){5}[0-9A-F]{2}) (.*)", line)
            if not match:
                continue
            mac, name = match.groups()
            if self.address and mac.lower() != self.address.lower():
                continue
            if self.address or _normalize_name(name) == wanted:
                self.device_address, self.device_name = mac, name
                info = self._bluetoothctl("info", mac, timeout=10)
                self.advertised_uuids = set(
                    re.findall(r"UUID:.*\(([0-9a-f-]{36})\)", info)
                )
                return mac
        return None

    def connect(self) -> None:
        """Connect over the LE bearer and subscribe to command responses.

        The robot advertises with its controller's public address; retry with
        a random address type in case a robot ever uses one.
        """
        if self.device_address is None:
            raise BleLinkError("discover() must find the robot before connect()")
        try:
            self._att = AttClient(
                self.device_address, address_type="public", timeout=self.scan_timeout
            )
            self._att.connect()
        except OSError:
            self._att = AttClient(
                self.device_address, address_type="random", timeout=self.scan_timeout
            )
            self._att.connect()
        self._command_char = self._att.characteristic(COMMAND_CHAR_UUID)
        self._response_char = self._att.characteristic(RESPONSE_CHAR_UUID)
        if self._command_char is None or self._response_char is None:
            raise BleLinkError(
                "connected, but the BLE command service characteristics are "
                f"missing (discovered: {[c.uuid for c in self._att.characteristics]})"
            )
        self._att.subscribe(self._response_char)

    def stop(self) -> None:
        """Close the LE connection (best effort)."""
        if self._att is not None:
            self._att.close()
            self._att = None

    # -- GATT helpers ---------------------------------------------------

    def _require_att(self) -> AttClient:
        if self._att is None:
            raise BleLinkError("not connected")
        return self._att

    def characteristic(self, uuid: str) -> Characteristic | None:
        """Return the discovered characteristic for a UUID, or None."""
        return self._require_att().characteristic(uuid)

    def read_char(self, uuid: str) -> str:
        """Read a characteristic by UUID and decode it as UTF-8."""
        att = self._require_att()
        char = att.characteristic(uuid)
        if char is None:
            raise BleLinkError(f"characteristic {uuid} not found")
        return att.read(char.value_handle).decode("utf-8", errors="replace")

    # -- command protocol ----------------------------------------------

    def command(self, cmd: str) -> str:
        """Send a command string and return the final response string.

        Synchronous firmware commands (PING, ECHO, JOURNAL_*, PIN_) place
        their result directly in the response characteristic, which is read
        back. Daemon-proxied commands ack with "OK: working" and deliver the
        real payload as a notification, which is awaited with a timeout.
        """
        att = self._require_att()
        assert self._command_char is not None and self._response_char is not None
        att.drain_notifications()
        att.write(self._command_char.value_handle, cmd.encode("utf-8"))
        response = att.read(self._response_char.value_handle).decode(
            "utf-8", errors="replace"
        )
        if response != self.ACK:
            return response
        try:
            payload = att.wait_notification(self.response_timeout)
        except TimeoutError:
            raise BleLinkError(
                f"No notification within {self.response_timeout}s for {cmd!r}"
            ) from None
        return payload.decode("utf-8", errors="replace")
