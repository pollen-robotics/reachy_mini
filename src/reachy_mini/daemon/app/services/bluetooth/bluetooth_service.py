#!/usr/bin/env python3
"""Bluetooth service for Reachy Mini using direct DBus API.

Includes a fixed NoInputNoOutput agent for automatic Just Works pairing.
"""
# mypy: ignore-errors

import fcntl
import hashlib
import json
import logging
import os
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

import dbus
import dbus.mainloop.glib
import dbus.service
from gi.repository import GLib

# ─── Hardware identity (inlined) ──────────────────────────────────────────
#
# IMPORTANT: this file runs on `/usr/bin/python3` (see the systemd unit in
# install_service_bluetooth.sh), NOT inside the daemon's venv. That means
# we CANNOT `from reachy_mini.utils.hardware_id import …` — the package
# is not installed in the system Python, and its own __init__ chains into
# numpy/scipy which also aren't there.
#
# The canonical implementation lives in `reachy_mini.utils.hardware_id`.
# Keep the constants and logic below in lock-step with it. The drift-check
# test at `tests/unit_tests/test_hardware_id_inline_consistency.py` parses
# both files as plain text and asserts the literals match — restore /
# update it whenever this block changes.
POLLEN_AUDIO_VID = "38fb"
POLLEN_AUDIO_PID = "1001"


def _read_raw_audio_serial() -> str | None:
    """Read the raw Pollen audio device USB serial from sysfs."""
    usb_root = Path("/sys/bus/usb/devices")
    if not usb_root.exists():
        return None
    for dev in usb_root.iterdir():
        try:
            if (dev / "idVendor").read_text().strip() != POLLEN_AUDIO_VID:
                continue
            if (dev / "idProduct").read_text().strip() != POLLEN_AUDIO_PID:
                continue
            serial = (dev / "serial").read_text().strip()
            return serial or None
        except (OSError, FileNotFoundError):
            continue
    return None


def get_hardware_id() -> str | None:
    """Public hardware ID — SHA-256 of the raw serial, truncated to 16 hex."""
    raw = _read_raw_audio_serial()
    if raw is None:
        return None
    return hashlib.sha256(raw.encode("ascii")).hexdigest()[:16]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service and Characteristic UUIDs
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
COMMAND_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef1"
RESPONSE_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef2"

# Device Information Service UUIDs (standard BLE service)
DEVICE_INFO_SERVICE_UUID = "0000180a-0000-1000-8000-00805f9b34fb"
MANUFACTURER_NAME_UUID = "00002a29-0000-1000-8000-00805f9b34fb"
MODEL_NUMBER_UUID = "00002a24-0000-1000-8000-00805f9b34fb"
FIRMWARE_REVISION_UUID = "00002a26-0000-1000-8000-00805f9b34fb"

# Custom Reachy Status Service UUIDs
REACHY_STATUS_SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef3"
NETWORK_STATUS_UUID = "12345678-1234-5678-1234-56789abcdef4"
SYSTEM_STATUS_UUID = "12345678-1234-5678-1234-56789abcdef5"
AVAILABLE_COMMANDS_UUID = "12345678-1234-5678-1234-56789abcdef6"
HARDWARE_ID_UUID = "12345678-1234-5678-1234-56789abcdef7"

BLUEZ_SERVICE_NAME = "org.bluez"
GATT_MANAGER_IFACE = "org.bluez.GattManager1"
DBUS_OM_IFACE = "org.freedesktop.DBus.ObjectManager"
DBUS_PROP_IFACE = "org.freedesktop.DBus.Properties"
GATT_SERVICE_IFACE = "org.bluez.GattService1"
GATT_CHRC_IFACE = "org.bluez.GattCharacteristic1"
GATT_DESC_IFACE = "org.bluez.GattDescriptor1"
LE_ADVERTISING_MANAGER_IFACE = "org.bluez.LEAdvertisingManager1"
LE_ADVERTISEMENT_IFACE = "org.bluez.LEAdvertisement1"
AGENT_PATH = "/org/bluez/agent"

# Descriptor UUIDs
USER_DESCRIPTION_UUID = "00002901-0000-1000-8000-00805f9b34fb"


# =======================
# BLE Agent for Just Works
# =======================
class NoInputAgent(dbus.service.Object):
    """BLE Agent for Just Works pairing."""

    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
    def Release(self, *args):
        """Handle release of the agent."""
        logger.info("Agent released")

    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="s")
    def RequestPinCode(self, *args):
        """Automatically provide an empty pin code for Just Works pairing."""
        logger.info(f"RequestPinCode called with args: {args}, returning empty")
        return ""

    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="u")
    def RequestPasskey(self, *args):
        """Automatically provide a passkey of 0 for Just Works pairing."""
        logger.info(f"RequestPasskey called with args: {args}, returning 0")
        return dbus.UInt32(0)

    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
    def RequestConfirmation(self, *args):
        """Automatically confirm the pairing request."""
        logger.info(
            f"RequestConfirmation called with args: {args}, accepting automatically"
        )
        return

    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
    def DisplayPinCode(self, *args):
        """Handle displaying the pin code (not used in Just Works)."""
        logger.info(f"DisplayPinCode called with args: {args}")

    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
    def DisplayPasskey(self, *args):
        """Handle displaying the passkey (not used in Just Works)."""
        logger.info(f"DisplayPasskey called with args: {args}")

    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
    def AuthorizeService(self, *args):
        """Handle service authorization requests."""
        logger.info(f"AuthorizeService called with args: {args}")

    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
    def Cancel(self, *args):
        """Handle cancellation of the agent request."""
        logger.info("Agent request canceled")


# =======================
# BLE Advertisement
# =======================
class Advertisement(dbus.service.Object):
    """BLE Advertisement."""

    PATH_BASE = "/org/bluez/advertisement"

    def __init__(self, bus, index, advertising_type, local_name):
        """Initialize the Advertisement."""
        self.path = self.PATH_BASE + str(index)
        self.bus = bus
        self.ad_type = advertising_type
        self.local_name = local_name
        self.service_uuids = None
        self.include_tx_power = False
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        """Return the properties of the advertisement."""
        props = {"Type": self.ad_type}
        if self.local_name:
            props["LocalName"] = dbus.String(self.local_name)
        if self.service_uuids:
            props["ServiceUUIDs"] = dbus.Array(self.service_uuids, signature="s")
        props["Appearance"] = dbus.UInt16(0x0000)
        props["Duration"] = dbus.UInt16(0)
        props["Timeout"] = dbus.UInt16(0)
        return {LE_ADVERTISEMENT_IFACE: props}

    def get_path(self):
        """Return the object path."""
        return dbus.ObjectPath(self.path)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
        """Return all properties of the advertisement."""
        if interface != LE_ADVERTISEMENT_IFACE:
            raise dbus.exceptions.DBusException(
                "org.freedesktop.DBus.Error.InvalidArgs",
                "Unknown interface " + interface,
            )
        return self.get_properties()[LE_ADVERTISEMENT_IFACE]

    @dbus.service.method(LE_ADVERTISEMENT_IFACE, in_signature="", out_signature="")
    def Release(self):
        """Handle release of the advertisement."""
        logger.info("Advertisement released")


# =======================
# BLE Characteristics & Service
# =======================
class Descriptor(dbus.service.Object):
    """GATT Descriptor."""

    def __init__(self, bus, index, uuid, flags, characteristic):
        """Initialize the Descriptor."""
        self.path = characteristic.path + "/desc" + str(index)
        self.bus = bus
        self.uuid = uuid
        self.flags = flags
        self.characteristic = characteristic
        self.value = []
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        """Return the properties of the descriptor."""
        return {
            GATT_DESC_IFACE: {
                "Characteristic": self.characteristic.get_path(),
                "UUID": self.uuid,
                "Flags": self.flags,
            }
        }

    def get_path(self):
        """Return the object path."""
        return dbus.ObjectPath(self.path)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
        """Return all properties of the descriptor."""
        if interface != GATT_DESC_IFACE:
            raise dbus.exceptions.DBusException(
                "org.freedesktop.DBus.Error.InvalidArgs", "Unknown interface"
            )
        return self.get_properties()[GATT_DESC_IFACE]

    @dbus.service.method(GATT_DESC_IFACE, in_signature="a{sv}", out_signature="ay")
    def ReadValue(self, options):
        """Handle read from the descriptor."""
        return dbus.Array(self.value, signature="y")

    @dbus.service.method(GATT_DESC_IFACE, in_signature="aya{sv}")
    def WriteValue(self, value, options):
        """Handle write to the descriptor."""
        self.value = value


class Characteristic(dbus.service.Object):
    """GATT Characteristic."""

    def __init__(self, bus, index, uuid, flags, service):
        """Initialize the Characteristic."""
        self.path = service.path + "/char" + str(index)
        self.bus = bus
        self.uuid = uuid
        self.service = service
        self.flags = flags
        self.value = []
        self.descriptors = []
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        """Return the properties of the characteristic."""
        props = {
            GATT_CHRC_IFACE: {
                "Service": self.service.get_path(),
                "UUID": self.uuid,
                "Flags": self.flags,
            }
        }
        if self.descriptors:
            props[GATT_CHRC_IFACE]["Descriptors"] = [
                d.get_path() for d in self.descriptors
            ]
        return props

    def get_path(self):
        """Return the object path."""
        return dbus.ObjectPath(self.path)

    def add_descriptor(self, descriptor):
        """Add a descriptor to this characteristic."""
        self.descriptors.append(descriptor)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
        """Return all properties of the characteristic."""
        if interface != GATT_CHRC_IFACE:
            raise dbus.exceptions.DBusException(
                "org.freedesktop.DBus.Error.InvalidArgs", "Unknown interface"
            )
        return self.get_properties()[GATT_CHRC_IFACE]

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="a{sv}", out_signature="ay")
    def ReadValue(self, options):
        """Handle read from the characteristic."""
        return dbus.Array(self.value, signature="y")

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="aya{sv}")
    def WriteValue(self, value, options):
        """Handle write to the characteristic."""
        self.value = value


class CommandCharacteristic(Characteristic):
    """Command Characteristic."""

    def __init__(self, bus, index, service, command_handler: Callable[[bytes], str]):
        """Initialize the Command Characteristic."""
        super().__init__(bus, index, COMMAND_CHAR_UUID, ["write"], service)
        self.command_handler = command_handler

    def WriteValue(self, value, options):
        """Handle write to the Command Characteristic."""
        command_bytes = bytes(value)
        response = self.command_handler(command_bytes)
        self.service.response_char.value = [
            dbus.Byte(b) for b in response.encode("utf-8")
        ]
        cmd_str = command_bytes.decode("utf-8", errors="replace").strip()
        if cmd_str.upper() not in ("JOURNAL_READ", "WIFI_STATUS"):
            logger.info(f"Command received: {response}")


class ResponseCharacteristic(Characteristic):
    """Response Characteristic."""

    def __init__(self, bus, index, service):
        """Initialize the Response Characteristic."""
        super().__init__(bus, index, RESPONSE_CHAR_UUID, ["read", "notify"], service)
        self.notifying = False

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="", out_signature="")
    def StartNotify(self):
        """Handle BlueZ notification subscription from a client."""
        self.notifying = True
        logger.info("Response notifications enabled")

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="", out_signature="")
    def StopNotify(self):
        """Handle BlueZ notification unsubscription from a client."""
        self.notifying = False
        logger.info("Response notifications disabled")
        # Stop journal streaming if running (client disconnected without JOURNAL_STOP)
        if hasattr(self.service, '_bt_service') and self.service._bt_service:
            self.service._bt_service._stop_journal()

    def send_notification(self, text: str):
        """Send a BLE notification with the given text."""
        self.value = [dbus.Byte(b) for b in text.encode("utf-8")]
        if self.notifying:
            self.PropertiesChanged(
                GATT_CHRC_IFACE, {"Value": dbus.Array(self.value, signature="y")}, []
            )

    @dbus.service.signal(DBUS_PROP_IFACE, signature="sa{sv}as")
    def PropertiesChanged(self, interface, changed, invalidated):
        """Emit PropertiesChanged signal for BLE notifications."""
        pass


class Service(dbus.service.Object):
    """GATT Service."""

    PATH_BASE = "/org/bluez/service"

    def __init__(
        self, bus, index, uuid, primary, command_handler: Callable[[bytes], str]
    ):
        """Initialize the GATT Service."""
        self.path = self.PATH_BASE + str(index)
        self.bus = bus
        self.uuid = uuid
        self.primary = primary
        self.characteristics = []
        dbus.service.Object.__init__(self, bus, self.path)
        # Response characteristic first
        self.response_char = ResponseCharacteristic(bus, 1, self)
        self.add_characteristic(self.response_char)
        # Command characteristic
        self.add_characteristic(CommandCharacteristic(bus, 0, self, command_handler))

    def get_properties(self):
        """Return the properties of the service."""
        return {
            GATT_SERVICE_IFACE: {
                "UUID": self.uuid,
                "Primary": self.primary,
                "Characteristics": [ch.get_path() for ch in self.characteristics],
            }
        }

    def get_path(self):
        """Return the object path."""
        return dbus.ObjectPath(self.path)

    def add_characteristic(self, ch):
        """Add a characteristic to the service."""
        self.characteristics.append(ch)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
        """Return all properties of the service."""
        if interface != GATT_SERVICE_IFACE:
            raise dbus.exceptions.DBusException(
                "org.freedesktop.DBus.Error.InvalidArgs", "Unknown interface"
            )
        return self.get_properties()[GATT_SERVICE_IFACE]


class StaticCharacteristic(Characteristic):
    """Read-only characteristic with static value."""

    def __init__(
        self, bus, index, uuid, service, value_str: str, description: str = None
    ):
        """Initialize the Static Characteristic."""
        super().__init__(bus, index, uuid, ["read"], service)
        self.value = [dbus.Byte(b) for b in value_str.encode("utf-8")]

        # Add user description descriptor if provided
        if description:
            desc = Descriptor(bus, 0, USER_DESCRIPTION_UUID, ["read"], self)
            desc.value = [dbus.Byte(b) for b in description.encode("utf-8")]
            self.add_descriptor(desc)


class DynamicCharacteristic(Characteristic):
    """Read-only characteristic with dynamically updatable value."""

    def __init__(
        self,
        bus,
        index,
        uuid,
        service,
        value_getter: Callable[[], str],
        description: str = None,
    ):
        """Initialize the Dynamic Characteristic."""
        super().__init__(bus, index, uuid, ["read"], service)
        self.value_getter = value_getter
        self.update_value()

        # Add user description descriptor if provided
        if description:
            desc = Descriptor(bus, 0, USER_DESCRIPTION_UUID, ["read"], self)
            desc.value = [dbus.Byte(b) for b in description.encode("utf-8")]
            self.add_descriptor(desc)

    def update_value(self):
        """Update the characteristic value from the getter function."""
        value_str = self.value_getter()
        self.value = [dbus.Byte(b) for b in value_str.encode("utf-8")]
        return True  # Keep periodic callback alive


class DeviceInfoService(dbus.service.Object):
    """Device Information Service."""

    PATH_BASE = "/org/bluez/device_info"

    def __init__(self, bus, index):
        """Initialize the Device Information Service."""
        self.path = self.PATH_BASE + str(index)
        self.bus = bus
        self.uuid = DEVICE_INFO_SERVICE_UUID
        self.primary = True
        self.characteristics = []
        dbus.service.Object.__init__(self, bus, self.path)

        # Get hotspot IP and format it
        hotspot_ip = get_hotspot_ip()
        firmware_value = f"[HOTSPOT]:{hotspot_ip}"

        # Add standard Device Info characteristics
        self.add_characteristic(
            StaticCharacteristic(
                bus, 0, MANUFACTURER_NAME_UUID, self, "Pollen Robotics"
            )
        )
        self.add_characteristic(
            StaticCharacteristic(bus, 1, MODEL_NUMBER_UUID, self, "Reachy Mini")
        )
        self.add_characteristic(
            StaticCharacteristic(bus, 2, FIRMWARE_REVISION_UUID, self, firmware_value)
        )

    def get_properties(self):
        """Return the properties of the service."""
        return {
            GATT_SERVICE_IFACE: {
                "UUID": self.uuid,
                "Primary": self.primary,
                "Characteristics": [ch.get_path() for ch in self.characteristics],
            }
        }

    def get_path(self):
        """Return the object path."""
        return dbus.ObjectPath(self.path)

    def add_characteristic(self, ch):
        """Add a characteristic to the service."""
        self.characteristics.append(ch)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
        """Return all properties of the service."""
        if interface != GATT_SERVICE_IFACE:
            raise dbus.exceptions.DBusException(
                "org.freedesktop.DBus.Error.InvalidArgs", "Unknown interface"
            )
        return self.get_properties()[GATT_SERVICE_IFACE]


class ReachyStatusService(dbus.service.Object):
    """Custom Reachy Status Service with network and system info."""

    PATH_BASE = "/org/bluez/reachy_status"

    def __init__(self, bus, index):
        """Initialize the Reachy Status Service."""
        self.path = self.PATH_BASE + str(index)
        self.bus = bus
        self.uuid = REACHY_STATUS_SERVICE_UUID
        self.primary = True
        self.characteristics = []
        dbus.service.Object.__init__(self, bus, self.path)

        # Get available commands (static)
        import os

        commands_dir = "commands"
        available_cmds = []
        if os.path.isdir(commands_dir):
            for f in os.listdir(commands_dir):
                if f.endswith(".sh"):
                    available_cmds.append(f.replace(".sh", ""))
        commands_value = ", ".join(available_cmds) if available_cmds else "None"

        # Add dynamic network status characteristic that auto-updates
        self.network_char = DynamicCharacteristic(
            bus, 0, NETWORK_STATUS_UUID, self, get_network_status, "Network Status"
        )
        self.add_characteristic(self.network_char)

        # Add static characteristics
        self.add_characteristic(
            StaticCharacteristic(
                bus, 1, SYSTEM_STATUS_UUID, self, "Online", "System Status"
            )
        )
        self.add_characteristic(
            StaticCharacteristic(
                bus,
                2,
                AVAILABLE_COMMANDS_UUID,
                self,
                commands_value,
                "Available Commands",
            )
        )

        # Hardware ID — robot-unique Pollen audio device serial. Read-only,
        # populated once at service init (the audio device is hot-pluggable in
        # principle but in practice present for the daemon's lifetime). Lives
        # here rather than the advertisement because the legacy 31-byte advert
        # is already at capacity.
        #
        # `get_hardware_id` is the locally-inlined version at the top of this
        # module — see the comment block there for why we can't import from
        # the venv-hosted `reachy_mini.utils.hardware_id`.
        self.add_characteristic(
            StaticCharacteristic(
                bus,
                3,
                HARDWARE_ID_UUID,
                self,
                get_hardware_id() or "unknown",
                "Hardware ID",
            )
        )

    def update_network_status(self):
        """Update the network status characteristic value."""
        if hasattr(self, "network_char"):
            self.network_char.update_value()
        return True  # Keep periodic callback alive

    def get_properties(self):
        """Return the properties of the service."""
        return {
            GATT_SERVICE_IFACE: {
                "UUID": self.uuid,
                "Primary": self.primary,
                "Characteristics": [ch.get_path() for ch in self.characteristics],
            }
        }

    def get_path(self):
        """Return the object path."""
        return dbus.ObjectPath(self.path)

    def add_characteristic(self, ch):
        """Add a characteristic to the service."""
        self.characteristics.append(ch)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
        """Return all properties of the service."""
        if interface != GATT_SERVICE_IFACE:
            raise dbus.exceptions.DBusException(
                "org.freedesktop.DBus.Error.InvalidArgs", "Unknown interface"
            )
        return self.get_properties()[GATT_SERVICE_IFACE]


class Application(dbus.service.Object):
    """GATT Application."""

    def __init__(self, bus, command_handler: Callable[[bytes], str]):
        """Initialize the GATT Application."""
        self.path = "/"
        self.services = []
        dbus.service.Object.__init__(self, bus, self.path)
        # Add command service
        self.services.append(Service(bus, 0, SERVICE_UUID, True, command_handler))
        # Add Device Information Service
        self.services.append(DeviceInfoService(bus, 1))
        # Add Custom Reachy Status Service
        self.reachy_status = ReachyStatusService(bus, 2)
        self.services.append(self.reachy_status)

    def get_path(self):
        """Return the object path."""
        return dbus.ObjectPath(self.path)

    @dbus.service.method(DBUS_OM_IFACE, out_signature="a{oa{sa{sv}}}")
    def GetManagedObjects(self):
        """Return a dictionary of all managed objects."""
        resp = {}
        for service in self.services:
            resp[service.get_path()] = service.get_properties()
            for ch in service.characteristics:
                resp[ch.get_path()] = ch.get_properties()
                # Include descriptors
                for desc in ch.descriptors:
                    resp[desc.get_path()] = desc.get_properties()
        return resp


# =======================
# Bluetooth Command Server
# =======================
class BluetoothCommandService:
    """Bluetooth Command Service."""

    # An authenticated session stays valid for this long after a successful
    # PIN, so a client can chain scan → connect → poll status without
    # re-authing — but a stale/abandoned session does NOT stay privileged
    # forever (the PIN is a short proximity secret printed on the device).
    SESSION_TTL_S = 300

    # Wrong-PIN throttle. The PIN is only a few characters, so unmetered
    # guessing over BLE would brute-force it quickly. The first
    # PIN_FREE_ATTEMPTS misses are free (fat-finger tolerance); every miss
    # after that locks the PIN_ command for an exponentially growing window
    # (PIN_LOCKOUT_BASE_S, doubling each time, capped at PIN_LOCKOUT_MAX_S).
    # The counter is robot-global and deliberately SURVIVES BLE disconnects
    # (see _on_central_disconnected) so an attacker cannot reset it by
    # reconnecting; a correct PIN clears it.
    PIN_FREE_ATTEMPTS = 3
    PIN_LOCKOUT_BASE_S = 5
    PIN_LOCKOUT_MAX_S = 300

    def __init__(self, device_name="ReachyMini", pin_code="00000"):
        """Initialize the Bluetooth Command Service."""
        self.device_name = device_name
        self.pin_code = pin_code
        self.connected = False
        # monotonic deadline for the TTL-bounded WiFi session (see _is_authed).
        self._authed_until = 0.0
        # Wrong-PIN throttle state (see PIN_* constants and _handle_command).
        # Both deliberately persist across disconnects so reconnecting does
        # not reset an in-progress lockout.
        self._pin_failures = 0
        self._pin_locked_until = 0.0
        self.bus = None
        self.app = None
        self.adv = None
        self.mainloop = None
        self._journal_proc = None
        self._journal_watch_id = None
        self._journal_buffer = ""
        # Advertising manager + the object path of the currently-connected
        # central, populated in start() / the disconnect watcher. Used to
        # re-assert advertising and reset session state when a central drops
        # (incl. ungraceful drops like an app crash) — see
        # _on_device_properties_changed.
        self._ad_manager = None
        self._connected_device_path = None

    def _start_journal(self) -> str:
        """Start journalctl -f and buffer output for poll-based reading."""
        if self._journal_proc is not None:
            return "OK: Journal already streaming"
        try:
            self._journal_buffer = ""
            self._journal_proc = subprocess.Popen(
                ["stdbuf", "-oL", "journalctl", "-f", "-n", "20", "--no-pager", "-u", "reachy-mini-daemon"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            # Set non-blocking so GLib IO watch doesn't block the main loop
            fd = self._journal_proc.stdout.fileno()
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

            self._journal_watch_id = GLib.io_add_watch(
                self._journal_proc.stdout,
                GLib.IO_IN | GLib.IO_HUP,
                self._on_journal_data,
            )
            logger.info("Journal streaming started")
            return "OK: Journal streaming started"
        except Exception as e:
            logger.error(f"Error starting journal: {e}")
            self._stop_journal()
            return f"ERROR: {e}"

    def _on_journal_data(self, source, condition):
        """GLib callback — accumulate journalctl output into the buffer."""
        if condition & GLib.IO_IN:
            try:
                data = source.read(4096)
                if data:
                    text = data.decode("utf-8", errors="replace")
                    self._journal_buffer += text
                    logger.info(f"Journal buffered: {len(text)} bytes, total: {len(self._journal_buffer)}")
                    # Cap buffer to ~32KB to avoid unbounded growth
                    if len(self._journal_buffer) > 32768:
                        self._journal_buffer = self._journal_buffer[-32768:]
            except BlockingIOError:
                pass
            except Exception as e:
                logger.error(f"Error reading journal: {e}")

        if condition & GLib.IO_HUP:
            logger.info("Journal process ended")
            self._stop_journal()
            return False

        return True

    def _read_journal(self) -> str:
        """Return buffered journal data and clear the buffer."""
        if self._journal_proc is None:
            return "ERROR: Journal not running"
        chunk = self._journal_buffer[:480]  # Stay within BLE limits
        self._journal_buffer = self._journal_buffer[480:]
        if chunk:
            logger.info(f"Journal read: {len(chunk)} bytes")
        return chunk if chunk else ""

    def _stop_journal(self):
        """Stop the journalctl streaming subprocess."""
        if self._journal_watch_id is not None:
            GLib.source_remove(self._journal_watch_id)
            self._journal_watch_id = None
        if self._journal_proc is not None:
            try:
                self._journal_proc.terminate()
                self._journal_proc.wait(timeout=2)
            except Exception:
                self._journal_proc.kill()
            self._journal_proc = None
            self._journal_buffer = ""
            logger.info("Journal streaming stopped")

    def _is_authed(self) -> bool:
        """Return whether the TTL-bounded authenticated session is still valid."""
        return self._authed_until > time.monotonic()

    def _pin_lockout_remaining(self) -> float:
        """Seconds left on the wrong-PIN lockout (0.0 if not locked).

        Touched only from the GLib mainloop thread (WriteValue and the
        disconnect handler both run there), so no lock is needed.
        """
        return max(0.0, self._pin_locked_until - time.monotonic())

    def _register_pin_failure(self) -> None:
        """Record a wrong PIN and arm the next lockout window.

        Misses beyond PIN_FREE_ATTEMPTS lock the PIN_ command for
        PIN_LOCKOUT_BASE_S, doubling per consecutive miss, capped at
        PIN_LOCKOUT_MAX_S.
        """
        self._pin_failures += 1
        over = self._pin_failures - self.PIN_FREE_ATTEMPTS
        if over > 0:
            delay = min(
                self.PIN_LOCKOUT_MAX_S,
                self.PIN_LOCKOUT_BASE_S * (2 ** (over - 1)),
            )
            self._pin_locked_until = time.monotonic() + delay
            logger.warning(
                f"Wrong PIN ({self._pin_failures} consecutive). "
                f"PIN locked for {delay}s."
            )

    def _reset_pin_throttle(self) -> None:
        """Clear the wrong-PIN counter and lockout after a successful auth."""
        self._pin_failures = 0
        self._pin_locked_until = 0.0

    def _emit_response(self, text: str) -> bool:
        """Push a result over the RESPONSE characteristic. Runs on the mainloop."""
        try:
            self.app.services[0].response_char.send_notification(text)
        except Exception as e:
            logger.error(f"Failed to emit BLE notification: {e}")
        return False  # GLib.idle_add: run once

    def _run_async(self, fn: Callable[[], str]) -> None:
        """Run a blocking daemon-proxy command off the BLE mainloop.

        The dbus/GLib mainloop must NEVER block: a synchronous 15s nmcli scan
        inside WriteValue would freeze advertising, journal streaming, and all
        other GATT I/O. We run the urllib work on a worker thread and marshal
        the result back onto the mainloop via GLib.idle_add → notification.
        The originating WriteValue returns an immediate "OK: working" ack; the
        client awaits the real result on the RESPONSE characteristic.
        """

        def worker() -> None:
            try:
                result = fn()
            except Exception as e:
                result = f"ERROR: {e}"
            GLib.idle_add(self._emit_response, result)

        threading.Thread(target=worker, daemon=True).start()

    def _handle_command(self, value: bytes) -> str:
        command_str = value.decode("utf-8").strip()
        upper = command_str.upper()
        # WIFI_STATUS and JOURNAL_READ are polled by clients; don't spam logs.
        if upper not in ("JOURNAL_READ", "WIFI_STATUS"):
            logger.info(f"Received command: {command_str}")
        # Custom command handling
        if command_str.upper() == "PING":
            return "PONG"
        elif command_str.upper() == "STATUS":
            # exec a "sudo ls" command and print the result
            try:
                result = subprocess.run(["sudo", "ls"], capture_output=True, text=True)
                logger.info(f"Command output: {result.stdout}")
            except Exception as e:
                logger.error(f"Error executing command: {e}")
            return "OK: System running"
        elif command_str.upper() == "JOURNAL_START":
            return self._start_journal()
        elif command_str.upper() == "JOURNAL_READ":
            return self._read_journal()
        elif command_str.upper() == "JOURNAL_STOP":
            self._stop_journal()
            return "OK: Journal streaming stopped"
        elif command_str.startswith("PIN_"):
            remaining = self._pin_lockout_remaining()
            if remaining > 0:
                # Locked out: reject WITHOUT comparing the PIN, so a correct
                # guess landed mid-spree doesn't win and the lockout window is
                # the real bottleneck. int()+1 rounds up so we never show "0s".
                return (
                    f"ERROR: Too many attempts. Try again in {int(remaining) + 1}s."
                )
            pin = command_str[4:].strip()
            if pin == self.pin_code:
                self._reset_pin_throttle()
                self.connected = True
                # Open a TTL-bounded session for the WiFi commands so the
                # client can chain scan → connect → status without re-auth.
                self._authed_until = time.monotonic() + self.SESSION_TTL_S
                return "OK: Connected"
            else:
                self._register_pin_failure()
                return "ERROR: Incorrect PIN"

        # Daemon software update over BLE. Proxies to the daemon's existing
        # /update/* API on localhost (no logic duplicated). Updating the
        # daemon is privileged, so all three require a live TTL session (the
        # same auth gate the WiFi commands use). They run OFF the mainloop via
        # _run_async and deliver the real result over the RESPONSE
        # notification; WriteValue returns the "OK: working" ack immediately.
        # NB: /update/start-from-ref (arbitrary git ref) is intentionally NOT
        # exposed over BLE — it's a much larger attack surface than updating
        # to the official published release.
        # CAVEAT: a successful update ends with `systemctl restart
        # reachy-mini-daemon`, which drops the daemon's HTTP server AND wipes
        # the in-memory job registry. So UPDATE_INFO polling cannot observe a
        # terminal DONE — it tails off into "Daemon unreachable" / "Unknown
        # job". Clients infer success by reconnecting and re-running
        # UPDATE_CHECK (current_version == latest), not by polling to the end.
        elif upper == "UPDATE_CHECK":
            if not self._is_authed():
                return "ERROR: Not connected. Please authenticate first."
            self._run_async(_update_check)
            return "OK: working"
        elif upper == "UPDATE_START":
            if not self._is_authed():
                return "ERROR: Not connected. Please authenticate first."
            self._run_async(_update_start)
            return "OK: working"
        elif upper.startswith("UPDATE_INFO "):
            if not self._is_authed():
                return "ERROR: Not connected. Please authenticate first."
            job_id = command_str[len("UPDATE_INFO ") :].strip()
            self._run_async(lambda: _update_info(job_id))
            return "OK: working"
        # WiFi provisioning over BLE. All of these proxy to the daemon and may
        # block (nmcli rescan ~10s), so they run OFF the mainloop via
        # _run_async and deliver their result over the RESPONSE notification;
        # WriteValue returns the "OK: working" ack immediately. Mutating
        # commands require a live TTL session (B2); WIFI_STATUS is public but
        # withholds the saved-network list unless authed (B3); the password in
        # WIFI_CONNECT_ENC is sealed end-to-end so it is never cleartext here.
        elif upper == "WIFI_KEYEX":
            self._run_async(_wifi_keyex)
            return "OK: working"
        elif upper == "WIFI_STATUS":
            authed = self._is_authed()
            self._run_async(lambda: _wifi_status(authed))
            return "OK: working"
        elif upper == "WIFI_SCAN":
            if not self._is_authed():
                return "ERROR: Not connected. Please authenticate first."
            self._run_async(_wifi_scan)
            return "OK: working"
        elif upper.startswith("WIFI_CONNECT_ENC "):
            if not self._is_authed():
                return "ERROR: Not connected. Please authenticate first."
            blob = command_str[len("WIFI_CONNECT_ENC ") :]
            self._run_async(lambda: _wifi_connect_sealed(blob))
            return "OK: working"
        elif upper.startswith("WIFI_FORGET "):
            if not self._is_authed():
                return "ERROR: Not connected. Please authenticate first."
            ssid = command_str[len("WIFI_FORGET ") :]
            self._run_async(lambda: _wifi_forget(ssid))
            return "OK: working"

        # else if command starts with "CMD_xxxxx" check if  commands directory contains the said named script command xxxx.sh and run its, show output or/and send to read
        elif command_str.startswith("CMD_"):
            if not self.connected:
                return "ERROR: Not connected. Please authenticate first."
            try:
                script_name = command_str[4:].strip() + ".sh"
                script_path = os.path.join("commands", script_name)
                if os.path.isfile(script_path):
                    try:
                        result = subprocess.run(
                            ["sudo", script_path], capture_output=True, text=True
                        )
                        logger.info(f"Command output: {result.stdout}")
                    except Exception as e:
                        logger.error(f"Error executing command: {e}")
                else:
                    return f"ERROR: Command '{script_name}' not found"
            except Exception as e:
                logger.error(f"Error processing command: {e}")
                return "ERROR: Command execution failed"
            finally:
                self.connected = False  # reset connection after command
        else:
            return f"ECHO: {command_str}"

    def start(self):
        """Start the Bluetooth Command Service."""
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        self.bus = dbus.SystemBus()

        # BLE Agent registration
        agent_manager = dbus.Interface(
            self.bus.get_object("org.bluez", "/org/bluez"), "org.bluez.AgentManager1"
        )
        self.agent = NoInputAgent(self.bus, AGENT_PATH)
        agent_manager.RegisterAgent(AGENT_PATH, "NoInputNoOutput")
        agent_manager.RequestDefaultAgent(AGENT_PATH)
        logger.info("BLE Agent registered for Just Works pairing")

        # Find adapter
        adapter = self._find_adapter()
        if not adapter:
            raise Exception("Bluetooth adapter not found")

        adapter_props = dbus.Interface(adapter, DBUS_PROP_IFACE)
        adapter_props.Set("org.bluez.Adapter1", "Powered", dbus.Boolean(True))
        adapter_props.Set("org.bluez.Adapter1", "Discoverable", dbus.Boolean(True))
        adapter_props.Set("org.bluez.Adapter1", "DiscoverableTimeout", dbus.UInt32(0))
        adapter_props.Set("org.bluez.Adapter1", "Pairable", dbus.Boolean(True))

        # Register GATT application
        service_manager = dbus.Interface(adapter, GATT_MANAGER_IFACE)
        self.app = Application(self.bus, self._handle_command)
        # Back-reference so ResponseCharacteristic can stop journal on disconnect
        self.app.services[0]._bt_service = self
        service_manager.RegisterApplication(
            self.app.get_path(),
            {},
            reply_handler=lambda: logger.info("GATT app registered"),
            error_handler=lambda e: logger.error(f"Failed to register GATT app: {e}"),
        )

        # Register advertisement
        ad_manager = dbus.Interface(adapter, LE_ADVERTISING_MANAGER_IFACE)
        self._ad_manager = ad_manager
        self.adv = Advertisement(self.bus, 0, "peripheral", self.device_name)
        # Only advertise main service UUID to avoid advertisement size limits
        # All services are still available when connected
        self.adv.service_uuids = [REACHY_STATUS_SERVICE_UUID]
        ad_manager.RegisterAdvertisement(
            self.adv.get_path(),
            {},
            reply_handler=lambda: logger.info("Advertisement registered"),
            error_handler=lambda e: logger.error(
                f"Failed to register advertisement: {e}"
            ),
        )

        # Watch for central connect/disconnect. BlueZ emits PropertiesChanged
        # on org.bluez.Device1 with Connected=true/false. We use the false
        # edge to clean up after a dropped client — crucially including
        # UNGRACEFUL drops (app crash), where StopNotify may fire but session
        # state isn't reset and advertising isn't re-asserted. The signal
        # fires when BlueZ reaps the link (at the supervision timeout for an
        # abrupt loss), so this makes the service deterministically reusable
        # the moment BlueZ reports the drop.
        self.bus.add_signal_receiver(
            self._on_device_properties_changed,
            dbus_interface=DBUS_PROP_IFACE,
            signal_name="PropertiesChanged",
            arg0="org.bluez.Device1",
            path_keyword="path",
        )

        # Setup periodic network status updates (every 10 seconds)
        GLib.timeout_add_seconds(10, self.app.reachy_status.update_network_status)

        logger.info(f"✓ Bluetooth service started as '{self.device_name}'")

    def _on_device_properties_changed(self, interface, changed, invalidated, path=None):
        """React to BlueZ Device1 connect/disconnect transitions."""
        if interface != "org.bluez.Device1" or "Connected" not in changed:
            return
        if bool(changed["Connected"]):
            self._connected_device_path = path
            logger.info(f"BLE central connected: {path}")
        else:
            logger.info(f"BLE central disconnected: {path}")
            # Only act on the device we tracked as connected, so a stale
            # disconnect signal can't clobber a client that just reconnected.
            if self._connected_device_path in (None, path):
                self._connected_device_path = None
                self._on_central_disconnected()

    def _on_central_disconnected(self):
        """Clean up after a central drops (graceful or crash).

        Resets the PIN/TTL session so a new client must re-authenticate,
        stops any journal stream the dropped client left running, and
        re-asserts advertising so the robot is immediately reconnectable
        rather than relying on BlueZ to auto-resume.

        The wrong-PIN throttle (_pin_failures / _pin_locked_until) is
        deliberately NOT reset here: otherwise an attacker could wipe an
        in-progress lockout just by dropping and re-opening the link.
        """
        self.connected = False
        self._authed_until = 0.0
        self._stop_journal()
        self._reassert_advertising()

    def _reassert_advertising(self):
        """Re-register the advertisement so the robot stays discoverable.

        Belt-and-suspenders: BlueZ usually resumes a registered connectable
        advert after a link drops, but that's version-dependent. Unregister
        (best-effort) then register again; both errors are non-fatal (an
        AlreadyExists on register just means it was still active).
        """
        if self._ad_manager is None or self.adv is None:
            return
        try:
            self._ad_manager.UnregisterAdvertisement(self.adv.get_path())
        except dbus.exceptions.DBusException:
            pass  # not currently registered — fine
        try:
            self._ad_manager.RegisterAdvertisement(
                self.adv.get_path(),
                {},
                reply_handler=lambda: logger.info("Advertisement re-asserted after disconnect"),
                error_handler=lambda e: logger.warning(
                    f"Re-assert advertisement failed (non-fatal): {e}"
                ),
            )
        except dbus.exceptions.DBusException as e:
            logger.warning(f"Re-assert advertisement raised (non-fatal): {e}")

    def _find_adapter(self):
        remote_om = dbus.Interface(
            self.bus.get_object(BLUEZ_SERVICE_NAME, "/"), DBUS_OM_IFACE
        )
        objects = remote_om.GetManagedObjects()
        for path, props in objects.items():
            if GATT_MANAGER_IFACE in props and LE_ADVERTISING_MANAGER_IFACE in props:
                return self.bus.get_object(BLUEZ_SERVICE_NAME, path)
        return None

    def run(self):
        """Run the Bluetooth Command Service."""
        self.start()
        self.mainloop = GLib.MainLoop()
        try:
            logger.info("Running. Press Ctrl+C to exit...")
            self.mainloop.run()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self._stop_journal()
            self.mainloop.quit()


def get_pin() -> str:
    """Last 5 chars of the Pollen audio device serial — used as BLE pairing PIN.

    The PIN is a proximity-only short-lived secret, so leaking 5 chars of
    the raw serial is acceptable; the public hardware ID surfaced over the
    GATT characteristic is hashed (see `get_hardware_id` above). Falls back
    to a fixed default when no Reachy is attached so pairing still works
    on a dev workstation.

    Mirrors `reachy_mini.utils.hardware_id.get_pin`; see the inline-import
    note at the top of this file for why we don't share the implementation.
    """
    default_pin = "46879"
    raw = _read_raw_audio_serial()
    if raw and len(raw) >= 5:
        return raw[-5:]
    return default_pin


def get_network_status() -> str:
    """Get comprehensive network status with mode detection.

    Returns formatted string: {MODE} [interface] address ; [interface] address
    MODE: HOTSPOT (wlan0 is 10.0.0.x), CONNECTED (has IPs), OFFLINE (no IPs)
    """
    try:
        # Get network interfaces and IPs using ifconfig
        result = subprocess.run(
            ["ip", "-4", "addr", "show"], capture_output=True, text=True
        )

        interfaces = {}
        current_interface = None

        for line in result.stdout.splitlines():
            line = line.strip()
            # Detect interface name (e.g., "2: wlan0: <BROADCAST...")
            if line and not line.startswith("inet"):
                parts = line.split(":")
                if len(parts) >= 2 and parts[1].strip():
                    # Extract interface name (skip loopback)
                    iface = parts[1].strip()
                    if iface != "lo":
                        current_interface = iface
            # Extract IP address
            elif line.startswith("inet ") and current_interface:
                inet_parts = line.split()
                if len(inet_parts) >= 2:
                    ip_with_mask = inet_parts[1]
                    ip_addr = ip_with_mask.split("/")[0]
                    interfaces[current_interface] = ip_addr

        # Determine mode
        mode = "OFFLINE"
        if interfaces:
            # Check if wlan0 has 10.42.0.1 address (hotspot mode)
            wlan0_ip = interfaces.get("wlan0", "")
            if wlan0_ip.startswith("10.42.0.1"):
                mode = "HOTSPOT"
            else:
                mode = "CONNECTED"

        # Format output: {MODE} [interface] address ; [interface] address
        if not interfaces:
            return "OFFLINE"

        interface_strings = [f"[{iface}] {ip}" for iface, ip in interfaces.items()]
        return f"{mode} {' ; '.join(interface_strings)}"

    except Exception as e:
        logger.error(f"Error getting network status: {e}")
        return "ERROR"


def get_hotspot_ip() -> str:
    """Get the hotspot IP address from network interfaces (legacy function)."""
    status = get_network_status()
    # Extract first IP for backwards compatibility
    if "[" in status and "]" in status:
        try:
            return status.split("]")[1].split(";")[0].strip()
        except (IndexError, AttributeError):
            return "0.0.0.0"
    return "0.0.0.0"


# =======================
# WiFi provisioning over BLE  (proxy to the daemon's /wifi/* routes)
# =======================
# This service runs under the SYSTEM python (see install_service_bluetooth.sh),
# not the daemon venv — so it stays stdlib-only (`urllib`) and does NO crypto.
# It relays opaque bytes to the daemon on localhost; the daemon owns nmcli AND
# all the X25519/AES-GCM work (it has `cryptography`). See the daemon route
# `wifi_config.connect_to_wifi_network_sealed` for the sealed-PSK scheme.
#
# The WiFi password is NEVER seen here in cleartext: the phone seals it and
# the daemon opens it. `WIFI_CONNECT_ENC` carries only ciphertext.

DAEMON_LOCAL_URL = "http://127.0.0.1:8000"
WIFI_HTTP_TIMEOUT_S = 4.0
WIFI_SCAN_HTTP_TIMEOUT_S = 15.0  # nmcli rescan is slow
# Bound the scan reply by serialized BYTES (not item count): a single
# notification must fit the negotiated ATT MTU. 180 bytes is safe even when
# the phone never negotiates above the ~185-byte default.
WIFI_SCAN_MTU_BUDGET = 180


def _daemon_request(
    method: str,
    path: str,
    params: "dict[str, str] | None" = None,
    data: "dict | None" = None,
    timeout: float = WIFI_HTTP_TIMEOUT_S,
):
    """Local HTTP request to the daemon; return parsed JSON (or str/None)."""
    url = DAEMON_LOCAL_URL + path
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    body = None
    headers = {}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, method=method, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return raw.decode("utf-8", errors="replace")


# --- Daemon software update proxy (reuses _daemon_request above) -------------
# Relays to the daemon's /update/* routes. /update/available hits PyPI, so
# these use a longer timeout than the WiFi calls. /update/start-from-ref
# (arbitrary git ref) is intentionally NOT proxied — far larger attack surface
# than the official published release.
_UPDATE_HTTP_TIMEOUT_S = 30.0
# Like _wifi_scan, an UPDATE_INFO reply must fit a single RESPONSE
# notification, so its whole serialized payload is bounded by bytes.
_UPDATE_MTU_BUDGET = 180


def _update_check() -> str:
    """Report whether a daemon update is available (compact JSON for BLE)."""
    try:
        data = _daemon_request(
            "GET",
            "/update/available",
            {"pre_release": "false"},
            timeout=_UPDATE_HTTP_TIMEOUT_S,
        )
        rm = (data or {}).get("update", {}).get("reachy_mini", {})
        compact = {
            "available": rm.get("is_available"),
            "current": rm.get("current_version"),
            "latest": rm.get("available_version"),
        }
        return json.dumps(compact, separators=(",", ":"), ensure_ascii=False)
    except urllib.error.HTTPError as e:
        if e.code == 400:
            return "ERROR: Update in progress"
        return "ERROR: Check failed"
    except urllib.error.URLError:
        return "ERROR: Daemon unreachable"
    except Exception as e:
        return f"ERROR: {e}"


def _wifi_status(authed: bool) -> str:
    """Compact WiFi state for a BLE read/notify.

    Shape: {"mode","connected","error"} plus {"known":[...]} ONLY when the
    session is authenticated. The saved-network list is an owner-location
    fingerprint, so it is never exposed to an unauthenticated peer.
    """
    try:
        status = _daemon_request("GET", "/wifi/status") or {}
        compact = {
            "mode": status.get("mode"),
            "connected": status.get("connected_network"),
        }
        if authed:
            compact["known"] = status.get("known_networks", [])
        err = _daemon_request("GET", "/wifi/error") or {}
        compact["error"] = err.get("error")
        return json.dumps(compact, separators=(",", ":"), ensure_ascii=False)
    except urllib.error.URLError:
        return json.dumps(
            {"mode": None, "connected": None, "error": "daemon_unreachable"},
            separators=(",", ":"),
        )
    except Exception as e:
        return json.dumps(
            {"mode": None, "connected": None, "error": str(e)},
            separators=(",", ":"),
        )


def _wifi_keyex() -> str:
    """Relay the daemon's ephemeral provisioning public key to the phone.

    Returns the daemon's {"kid","pk","alg"} JSON verbatim (~110 bytes, fits a
    single MTU). The phone uses `pk` to seal the PSK; see _wifi_connect_sealed.
    """
    try:
        data = _daemon_request("GET", "/wifi/prov_key")
        return json.dumps(data, separators=(",", ":"))
    except urllib.error.URLError:
        return "ERROR: Daemon unreachable"
    except Exception as e:
        return f"ERROR: {e}"


def _update_start() -> str:
    """Trigger the daemon update (latest published release). Returns the job id."""
    try:
        data = _daemon_request(
            "POST",
            "/update/start",
            {"pre_release": "false"},
            timeout=_UPDATE_HTTP_TIMEOUT_S,
        )
        job_id = (data or {}).get("job_id") if isinstance(data, dict) else None
        if not job_id:
            return "ERROR: Start failed"
        # UPDATE_INFO <job_id> follows early progress, but the daemon restarts
        # on success and the job is then gone — see the CAVEAT in _handle_command.
        return f"OK: Update started {job_id}"
    except urllib.error.HTTPError as e:
        # The daemon returns 400 for "No update available" / "already in progress".
        if e.code == 400:
            return "ERROR: No update available or already in progress"
        return "ERROR: Start failed"
    except urllib.error.URLError:
        return "ERROR: Daemon unreachable"
    except Exception as e:
        return f"ERROR: {e}"


def _wifi_scan() -> str:
    """Scan for SSIDs. JSON array bounded by serialized byte length, or ERROR."""
    try:
        ssids = _daemon_request(
            "POST", "/wifi/scan_and_list", timeout=WIFI_SCAN_HTTP_TIMEOUT_S
        )
        if not isinstance(ssids, list):
            return json.dumps([])
        out: "list[str]" = []
        seen: "set[str]" = set()
        for s in ssids:
            if not isinstance(s, str) or not s or s in seen:
                continue
            seen.add(s)
            trial = out + [s]
            encoded = json.dumps(trial, separators=(",", ":"), ensure_ascii=False)
            if len(encoded.encode("utf-8")) > WIFI_SCAN_MTU_BUDGET:
                break
            out = trial
        return json.dumps(out, separators=(",", ":"), ensure_ascii=False)
    except urllib.error.HTTPError as e:
        if e.code == 409:
            return "ERROR: Busy"
        return "ERROR: Scan failed"
    except urllib.error.URLError:
        return "ERROR: Daemon unreachable"
    except Exception as e:
        return f"ERROR: {e}"


def _update_info(job_id: str) -> str:
    """Report the status of an update job (compact; full logs stay off BLE)."""
    job_id = job_id.strip()
    if not job_id:
        return "ERROR: Missing job_id"
    try:
        data = _daemon_request(
            "GET", "/update/info", {"job_id": job_id}, timeout=_UPDATE_HTTP_TIMEOUT_S
        )
        if not isinstance(data, dict):
            return "ERROR: Unknown job"
        logs = data.get("logs") or []
        # Forward only the last log line (the full log can be large; use the
        # daemon's /update/ws/logs websocket for the live stream). A single
        # RESPONSE notification must fit the negotiated ATT MTU, so bound the
        # WHOLE serialized payload by bytes and trim `last` — the one unbounded
        # field — until it fits. A fixed char cap is not enough: the JSON
        # envelope ({"status",...,"lines",...}) pushes the total over the MTU.
        compact = {"status": data.get("status"), "lines": len(logs), "last": ""}
        last = logs[-1] if logs else ""
        while True:
            compact["last"] = last
            encoded = json.dumps(compact, separators=(",", ":"), ensure_ascii=False)
            if len(encoded.encode("utf-8")) <= _UPDATE_MTU_BUDGET or not last:
                return encoded
            last = last[:-1]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "ERROR: Unknown job"
        return "ERROR: Info failed"
    except urllib.error.URLError:
        return "ERROR: Daemon unreachable"
    except Exception as e:
        return f"ERROR: {e}"


def _wifi_connect_sealed(blob: str) -> str:
    """Relay a sealed connect blob to the daemon.

    `blob` is the phone's JSON {"ssid","kid","epk","nonce","ct"} — all opaque
    to this service (the password is sealed; only the daemon can open it).
    """
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return "ERROR: Invalid payload (expected JSON)"
    for field in ("ssid", "kid", "epk", "nonce", "ct"):
        if not isinstance(data.get(field), str) or not data[field]:
            return f"ERROR: Missing field {field}"
    try:
        # Clear any stale error so the client can observe THIS attempt.
        try:
            _daemon_request("POST", "/wifi/reset_error")
        except Exception:
            pass  # non-fatal; logged daemon-side
        _daemon_request("POST", "/wifi/connect_sealed", data=data)
        return f"OK: Connecting to {data['ssid']}"
    except urllib.error.HTTPError as e:
        if e.code == 400:
            return "ERROR: Bad credentials (wrong PIN?)"
        if e.code == 409:
            return "ERROR: Busy"
        return "ERROR: Connect request failed"
    except urllib.error.URLError:
        return "ERROR: Daemon unreachable"
    except Exception as e:
        return f"ERROR: {e}"


def _wifi_forget(ssid: str) -> str:
    """Forget a saved WiFi network (daemon falls back to hotspot if active)."""
    ssid = ssid.strip()
    if not ssid:
        return "ERROR: Missing ssid"
    try:
        _daemon_request("POST", "/wifi/forget", params={"ssid": ssid})
        return f"OK: Forgotten {ssid}"
    except urllib.error.HTTPError as e:
        if e.code == 400:
            return "ERROR: Cannot forget hotspot"
        if e.code == 404:
            return "ERROR: Unknown ssid"
        if e.code == 409:
            return "ERROR: Busy"
        return "ERROR: Forget failed"
    except urllib.error.URLError:
        return "ERROR: Daemon unreachable"
    except Exception as e:
        return f"ERROR: {e}"


# =======================
# Main
# =======================
def main():
    """Run the Bluetooth Command Service."""
    pin = get_pin()

    bt_service = BluetoothCommandService(device_name="ReachyMini", pin_code=pin)
    bt_service.run()


if __name__ == "__main__":
    main()
