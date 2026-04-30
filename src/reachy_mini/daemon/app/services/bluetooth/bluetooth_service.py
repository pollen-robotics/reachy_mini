#!/usr/bin/env python3
"""Bluetooth service for Reachy Mini using direct DBus API.

Includes a fixed NoInputNoOutput agent for automatic Just Works pairing.
"""
# mypy: ignore-errors

import concurrent.futures
import fcntl
import json
import logging
import os
import socket
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

import dbus
import dbus.mainloop.glib
import dbus.service
from gi.repository import GLib

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
# install_id (UUID4 hex, generated once on first daemon boot). Same value
# also surfaces via mDNS TXT, the central listing's ``meta``, and
# ``GET /api/daemon/identity`` - so the mobile robot registry can merge
# this BLE row with the same robot's other sightings on this stable key.
INSTALL_ID_UUID = "12345678-1234-5678-1234-56789abcdef7"
ROBOT_NAME_UUID = "12345678-1234-5678-1234-56789abcdef8"

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
        self.manufacturer_data = None
        self.include_tx_power = False
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        """Return the properties of the advertisement.

        Note on payload size: legacy BLE advertisements are capped at
        31 bytes total. We deliberately omit ``ServiceUUIDs`` from the
        primary advertisement and rely on ``LocalName`` matching on
        the client side: a 128-bit UUID alone eats 18 bytes, which
        leaves no room for the ManufacturerData IPv4 payload (5 bytes
        per address). The full GATT service tree is still discoverable
        once the client connects, so this is purely an advertising-time
        size optimisation. ``Appearance=0x0000`` ("Unknown") is also
        skipped because it is uninformative and adds 4 bytes.
        """
        props = {"Type": self.ad_type}
        if self.local_name:
            props["LocalName"] = dbus.String(self.local_name)
        if self.manufacturer_data:
            props["ManufacturerData"] = dbus.Dictionary(
                self.manufacturer_data, signature="qv"
            )
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
        if hasattr(self.service, "_bt_service") and self.service._bt_service:
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

        # Identity characteristics (install_id + current robot name).
        # Resolved lazily through the daemon's local HTTP endpoint so
        # they keep up with rename + first-boot persistence without us
        # having to wire a second IPC channel into this DBus-only
        # service. ``Dynamic`` is fine here: the read frequency is
        # tiny (once at scan time on the mobile side).
        self.add_characteristic(
            DynamicCharacteristic(
                bus, 3, INSTALL_ID_UUID, self, _get_install_id, "Install ID"
            )
        )
        self.add_characteristic(
            DynamicCharacteristic(
                bus, 4, ROBOT_NAME_UUID, self, _get_robot_name, "Robot Name"
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

    def __init__(self, device_name="ReachyMini", pin_code="00000"):
        """Initialize the Bluetooth Command Service."""
        self.device_name = device_name
        self.pin_code = pin_code
        self.connected = False
        self.bus = None
        self.app = None
        self.adv = None
        self.mainloop = None
        self._journal_proc = None
        self._journal_watch_id = None
        self._journal_buffer = ""

    def _start_journal(self) -> str:
        """Start journalctl -f and buffer output for poll-based reading."""
        if self._journal_proc is not None:
            return "OK: Journal already streaming"
        try:
            self._journal_buffer = ""
            self._journal_proc = subprocess.Popen(
                [
                    "stdbuf",
                    "-oL",
                    "journalctl",
                    "-f",
                    "-n",
                    "20",
                    "--no-pager",
                    "-u",
                    "reachy-mini-daemon",
                ],
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
                    logger.info(
                        f"Journal buffered: {len(text)} bytes, total: {len(self._journal_buffer)}"
                    )
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

    def _handle_command(self, value: bytes) -> str:
        command_str = value.decode("utf-8").strip()
        upper = command_str.upper()
        # WIFI_STATUS and JOURNAL_READ are polled by clients; don't spam logs.
        if upper not in ("JOURNAL_READ", "WIFI_STATUS"):
            logger.info(f"Received command: {command_str}")
        # Custom command handling
        if upper == "PING":
            return "PONG"
        elif upper == "STATUS":
            # exec a "sudo ls" command and print the result
            try:
                result = subprocess.run(["sudo", "ls"], capture_output=True, text=True)
                logger.info(f"Command output: {result.stdout}")
            except Exception as e:
                logger.error(f"Error executing command: {e}")
            return "OK: System running"
        elif upper == "JOURNAL_START":
            return self._start_journal()
        elif upper == "JOURNAL_READ":
            return self._read_journal()
        elif upper == "JOURNAL_STOP":
            self._stop_journal()
            return "OK: Journal streaming stopped"
        elif command_str.startswith("PIN_"):
            pin = command_str[4:].strip()
            if pin == self.pin_code:
                self.connected = True
                return "OK: Connected"
            else:
                return "ERROR: Incorrect PIN"

        # WiFi provisioning commands. WIFI_STATUS and WIFI_PROBE are public
        # (read-only snapshots); mutating commands require prior PIN
        # authentication. Unlike CMD_*, we do NOT reset `self.connected`
        # afterwards so a client can chain scan -> connect -> poll status in a
        # single provisioning session.
        elif upper == "WIFI_STATUS":
            return _wifi_status()
        elif upper == "WIFI_PROBE":
            return _wifi_probe()
        elif upper == "WIFI_SCAN":
            if not self.connected:
                return "ERROR: Not connected. Please authenticate first."
            return _wifi_scan()
        elif upper.startswith("WIFI_CONNECT "):
            if not self.connected:
                return "ERROR: Not connected. Please authenticate first."
            payload = command_str[len("WIFI_CONNECT ") :]
            return _wifi_connect(payload)
        elif upper.startswith("WIFI_FORGET "):
            if not self.connected:
                return "ERROR: Not connected. Please authenticate first."
            ssid = command_str[len("WIFI_FORGET ") :]
            return _wifi_forget(ssid)

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
        self.adv = Advertisement(self.bus, 0, "peripheral", self.device_name)
        self.adv.service_uuids = [REACHY_STATUS_SERVICE_UUID]
        # Pull install_id + central_peer_id once at boot through the
        # identity endpoint that the GATT install_id characteristic
        # also reads. The advertis re-published on every refresh tick (see
        # :meth:`_refresh_network_info`) so a daemon that boots before
        # ``/api/daemon/identity`` is reachable still ends up with the
        # correct payload.
        identity = _fetch_identity()
        self.adv.manufacturer_data = encode_advert_manufacturer_data(
            identity.get("install_id", ""),
            identity.get("central_peer_id", ""),
            _current_network_mode_byte(),
        )
        self._ad_manager = ad_manager
        ad_manager.RegisterAdvertisement(
            self.adv.get_path(),
            {},
            reply_handler=lambda: logger.info("Advertisement registered"),
            error_handler=lambda e: logger.error(
                f"Failed to register advertisement: {e}"
            ),
        )

        # Refresh both GATT network characteristic and advertisement payload
        GLib.timeout_add_seconds(10, self._refresh_network_info)

        logger.info(f"✓ Bluetooth service started as '{self.device_name}'")

    def _refresh_network_info(self):
        """Periodic callback: update GATT characteristic and advertisement.

        Re-publishes the BLE advert when any of the three TLV
        payloads changes:

          - install_id (stable; the very first ticks may run before
            ``/api/daemon/identity`` is reachable, hence the upgrade)
          - central_peer_id (volatile; rotates on every relay
            reconnect - this tick is the cadence at which mobile
            scanners see the new peerId)
          - network_mode (dynamic; flips between OFFLINE / HOTSPOT /
            CONNECTED as the user joins a Wi-Fi or the daemon falls
            back to its own AP). The 10s tick is fast enough that a
            mobile scanner sees the transition before the user gives
            up retrying.
        """
        self.app.reachy_status.update_network_status()

        identity = _fetch_identity()
        new_data = encode_advert_manufacturer_data(
            identity.get("install_id", ""),
            identity.get("central_peer_id", ""),
            _current_network_mode_byte(),
        )
        if new_data != self.adv.manufacturer_data:
            self.adv.manufacturer_data = new_data
            try:
                self._ad_manager.UnregisterAdvertisement(self.adv.get_path())
                self._ad_manager.RegisterAdvertisement(
                    self.adv.get_path(),
                    {},
                    reply_handler=lambda: logger.info(
                        "Advertisement re-registered with updated install_id payload"
                    ),
                    error_handler=lambda e: logger.error(
                        f"Failed to re-register advertisement: {e}"
                    ),
                )
            except dbus.exceptions.DBusException as e:
                logger.warning(f"Could not refresh advertisement: {e}")

        return True

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


POLLEN_MANUFACTURER_ID = 0xFFFF  # Reserved ID for development/testing

# Versioned ManufacturerData layout we publish in the advertisement:
#
#   byte 0      : format version (currently 0x02)
#   bytes 1..N  : TLV blocks  -  [tag][len][value]
#
# Tag values:
#   0x01  install_id_prefix : len=8 bytes, the first 8 bytes of the
#         daemon's install_id (UUID4 hex). Used by mobile clients to
#         dedupe a BLE row against the same physical robot's localhost
#         loopback row + (eventually) its central listing without
#         having to GATT connect first (a connect breaks the scan +
#         needs pairing).
#   0x02  central_peer_id_prefix : len=8 bytes, the first 8 bytes of
#         the relay-assigned central peerId we got back on the
#         welcome frame. Volatile (rotates on every relay reconnect)
#         but used by mobile clients as a fallback dedup key for the
#         BLE row vs the central listing while the central server
#         does not yet propagate ``meta.install_id``. Absent when the
#         relay is offline / the daemon has no token.
#   0x03  network_mode : len=1 byte enum.
#           0x00 = OFFLINE   - no usable IP on any interface.
#           0x01 = HOTSPOT   - daemon is broadcasting its own AP
#                              (wlan0 == 10.42.0.1).
#           0x02 = CONNECTED - daemon has a real LAN/WAN IP (Wi-Fi
#                              joined OR USB-tether interface up).
#         Reads off ``ip -4 addr`` directly, so it reflects the
#         daemon's *local* network state - independent of whether
#         HF central / Internet are reachable. Mobile clients use
#         this to render an authoritative "Setup pending" badge
#         without false positives caused by transient relay flaps
#         (cf. tag 0x02 which depends on 6+ conditions and flickers
#         when central is unstable). Always emitted - we send 0x00
#         OFFLINE on subprocess failure rather than dropping the
#         TLV, so absence = "old daemon, no support".
#
# We deliberately drop the legacy IPv4 list from the advert: the
# 31-byte legacy advertising payload is essentially full once
# Flags + LocalName ("reachy-mini" = 13B) are accounted for, and
# every existing client reads IPs via the GATT NETWORK_STATUS
# characteristic anyway.
ADVERT_FORMAT_VERSION = 0x02
ADVERT_TLV_INSTALL_ID = 0x01
ADVERT_TLV_CENTRAL_PEER_ID = 0x02
ADVERT_TLV_NETWORK_MODE = 0x03
ADVERT_INSTALL_ID_PREFIX_BYTES = 8
ADVERT_CENTRAL_PEER_ID_PREFIX_BYTES = 8

# 1-byte enum values for ADVERT_TLV_NETWORK_MODE.
ADVERT_NETWORK_MODE_OFFLINE = 0x00
ADVERT_NETWORK_MODE_HOTSPOT = 0x01
ADVERT_NETWORK_MODE_CONNECTED = 0x02


# =======================
# WiFi provisioning over BLE
# =======================
# The Bluetooth service runs as its own systemd unit, separate from the FastAPI
# daemon. For WiFi provisioning we simply proxy BLE commands over localhost HTTP
# to the daemon's existing `/wifi/*` routes. This keeps the logic DRY (no
# duplicated `nmcli` plumbing) and reuses the daemon's `busy_lock`, threading
# and hotspot-fallback behavior.
#
# We stick to the Python stdlib (`urllib`) on purpose: the BT service uses the
# system Python, not the daemon's venv, so we can't assume `requests` is
# installed.

DAEMON_LOCAL_URL = "http://127.0.0.1:8000"
WIFI_HTTP_TIMEOUT_S = 4.0
WIFI_SCAN_HTTP_TIMEOUT_S = 15.0  # nmcli rescan is slow
WIFI_SCAN_MAX_RESULTS = 12  # keep payload inside a single BLE MTU
IDENTITY_HTTP_TIMEOUT_S = 1.5  # called from a GATT read; must not block

# Cache so a burst of GATT reads (mobile clients tend to pre-warm
# every characteristic on connect) doesn't hammer the local daemon.
# Values are immutable across the daemon's lifetime (install_id
# definitionally so; robot_name is mutable but a 5s lag during a
# rename is harmless because the mobile re-fetches via HTTP anyway).
_identity_cache: dict[str, str | float] = {
    "install_id": "",
    "robot_name": "",
    "central_peer_id": "",
    "ts": 0.0,
}
# central_peer_id rotates on every relay reconnect, so don't cache it
# longer than the BLE refresh tick can react to it.
_IDENTITY_CACHE_TTL_S = 5.0


def _fetch_identity() -> dict[str, str]:
    """Return a fresh ``{install_id, robot_name, central_peer_id}`` dict.

    Falls back to empty strings on any error (daemon down, route 404
    on a pre-revision-3 daemon, ...). GATT reads must never raise.
    """
    import time

    now = time.monotonic()
    if now - float(_identity_cache.get("ts", 0.0)) < _IDENTITY_CACHE_TTL_S:
        return {
            "install_id": str(_identity_cache.get("install_id", "")),
            "robot_name": str(_identity_cache.get("robot_name", "")),
            "central_peer_id": str(_identity_cache.get("central_peer_id", "")),
        }

    install_id = ""
    robot_name = ""
    central_peer_id = ""
    try:
        payload = _daemon_request(
            "GET", "/api/daemon/identity", timeout=IDENTITY_HTTP_TIMEOUT_S
        )
        if isinstance(payload, dict):
            install_id = str(payload.get("install_id") or "")
            robot_name = str(payload.get("robot_name") or "")
            central_peer_id = str(payload.get("central_peer_id") or "")
    except Exception as e:
        logger.debug(f"identity fetch failed (returning blanks): {e}")

    _identity_cache["install_id"] = install_id
    _identity_cache["robot_name"] = robot_name
    _identity_cache["central_peer_id"] = central_peer_id
    _identity_cache["ts"] = now
    return {
        "install_id": install_id,
        "robot_name": robot_name,
        "central_peer_id": central_peer_id,
    }


def _get_install_id() -> str:
    """GATT getter: return the install_id as a UTF-8 string (or "")."""
    return _fetch_identity()["install_id"]


def _get_robot_name() -> str:
    """GATT getter: return the human-readable robot_name as UTF-8 (or "")."""
    return _fetch_identity()["robot_name"]


def _daemon_request(
    method: str,
    path: str,
    params: dict[str, str] | None = None,
    timeout: float = WIFI_HTTP_TIMEOUT_S,
):
    """Perform a local HTTP request against the daemon and return parsed JSON (or None)."""
    url = DAEMON_LOCAL_URL + path
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        if not body:
            return None
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return body.decode("utf-8", errors="replace")


def _wifi_status() -> str:
    """Return the current WiFi state as compact JSON suitable for a BLE read.

    Shape: {"mode": str, "connected": str|null, "known": [str], "error": str|null}
    """
    try:
        status = _daemon_request("GET", "/wifi/status") or {}
        error_payload = _daemon_request("GET", "/wifi/error") or {}
        compact = {
            "mode": status.get("mode"),
            "connected": status.get("connected_network"),
            "known": status.get("known_networks", []),
            "error": error_payload.get("error"),
        }
        return json.dumps(compact, separators=(",", ":"), ensure_ascii=False)
    except urllib.error.URLError as e:
        logger.warning(f"wifi_status: daemon unreachable: {e}")
        return json.dumps(
            {
                "mode": None,
                "connected": None,
                "known": [],
                "error": "daemon_unreachable",
            },
            separators=(",", ":"),
        )
    except Exception as e:
        logger.exception("wifi_status failed")
        return json.dumps(
            {"mode": None, "connected": None, "known": [], "error": str(e)},
            separators=(",", ":"),
        )


def _wifi_scan() -> str:
    """Scan for nearby SSIDs. Returns a JSON array (top N) or an `ERROR:` string."""
    try:
        ssids = _daemon_request(
            "POST", "/wifi/scan_and_list", timeout=WIFI_SCAN_HTTP_TIMEOUT_S
        )
        if not isinstance(ssids, list):
            return json.dumps([])
        cleaned: list[str] = []
        seen: set[str] = set()
        for s in ssids:
            if isinstance(s, str) and s and s not in seen:
                seen.add(s)
                cleaned.append(s)
                if len(cleaned) >= WIFI_SCAN_MAX_RESULTS:
                    break
        return json.dumps(cleaned, separators=(",", ":"), ensure_ascii=False)
    except urllib.error.HTTPError as e:
        if e.code == 409:
            return "ERROR: Busy"
        logger.warning(f"wifi_scan HTTP error: {e}")
        return "ERROR: Scan failed"
    except urllib.error.URLError as e:
        logger.warning(f"wifi_scan: daemon unreachable: {e}")
        return "ERROR: Daemon unreachable"
    except Exception as e:
        logger.exception("wifi_scan failed")
        return f"ERROR: {e}"


def _wifi_connect(payload: str) -> str:
    """Kick off a connect attempt. `payload` is a JSON string: {"ssid": "...", "psk": "..."}.

    Returns immediately (the daemon runs the actual `nmcli` work on a thread).
    Clients should poll `WIFI_STATUS` to observe the outcome.
    """
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return "ERROR: Invalid payload (expected JSON)"

    ssid = data.get("ssid")
    psk = data.get("psk") or data.get("password") or ""
    if not isinstance(ssid, str) or not ssid:
        return "ERROR: Missing ssid"
    if not isinstance(psk, str):
        return "ERROR: Invalid psk"

    try:
        # Clear any stale error so the client can observe THIS attempt via /wifi/error.
        try:
            _daemon_request("POST", "/wifi/reset_error")
        except Exception:
            pass  # non-fatal
        _daemon_request("POST", "/wifi/connect", params={"ssid": ssid, "password": psk})
        return f"OK: Connecting to {ssid}"
    except urllib.error.HTTPError as e:
        if e.code == 409:
            return "ERROR: Busy"
        logger.warning(f"wifi_connect HTTP error: {e}")
        return "ERROR: Connect request failed"
    except urllib.error.URLError as e:
        logger.warning(f"wifi_connect: daemon unreachable: {e}")
        return "ERROR: Daemon unreachable"
    except Exception as e:
        logger.exception("wifi_connect failed")
        return f"ERROR: {e}"


def _wifi_forget(ssid: str) -> str:
    """Forget a saved WiFi network. Falls back to hotspot server-side if needed."""
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
        logger.warning(f"wifi_forget HTTP error: {e}")
        return "ERROR: Forget failed"
    except urllib.error.URLError as e:
        logger.warning(f"wifi_forget: daemon unreachable: {e}")
        return "ERROR: Daemon unreachable"
    except Exception as e:
        logger.exception("wifi_forget failed")
        return f"ERROR: {e}"


# =======================
# WiFi reachability probe
# =======================
#
# Runs entirely from the robot's point of view. The mobile app calls
# this once after ``WIFI_STATUS`` reports ``mode=wlan`` so it can tell
# the user *why* the robot is or is not actually online (vs. just
# blindly polling /api/daemon/status from the phone, which only works
# if the phone happens to share the WiFi).
#
# Hard rule: NO TLS in any of these probes. The Pi has no RTC; on a
# fresh boot it can sit at 1970-01-01 until ``systemd-timesyncd``
# catches up via the freshly-joined network. An HTTPS check would
# spuriously fail certificate validation in that window and lie to the
# user about being offline. Plain TCP / plain DNS sidesteps that
# entirely - we only care that packets reach the right places, not
# that the cert chain is healthy.
#
# Each individual probe is bounded by a short timeout, all four run in
# parallel, and the whole call returns within
# ``_PROBE_TOTAL_TIMEOUT_S`` so the BT command loop never stalls.

# 1.1.1.1 (Cloudflare) and one.one.one.one are deliberately chosen:
#  - hostnames are short, ASCII, and resolve fast
#  - both AS13335 endpoints have global anycast PoPs (low jitter)
#  - 443/tcp is the most common port that's *not* MITM'd by guest WiFi
_PROBE_DNS_HOST = "one.one.one.one"
_PROBE_INTERNET_HOST = "1.1.1.1"
_PROBE_INTERNET_PORT = 443
_PROBE_DNS_TIMEOUT_S = 1.5
_PROBE_TCP_TIMEOUT_S = 1.5
_PROBE_DAEMON_TIMEOUT_S = 1.0
# Total budget for the whole WIFI_PROBE call. With 4 parallel checks
# capped at 1.5s each, 2.5s is plenty of slack.
_PROBE_TOTAL_TIMEOUT_S = 2.5


def _probe_default_gateway_ip() -> str | None:
    """Return the IPv4 default gateway from ``/proc/net/route``, if any.

    Pure /proc parsing - no external command, no DNS, no network call.
    Returns ``None`` when there's no default route (= robot has no LAN
    yet, even if it has a link-local IP).
    """
    try:
        with open("/proc/net/route", "r", encoding="ascii") as f:
            next(f, None)  # skip header
            for line in f:
                fields = line.split()
                if len(fields) < 11:
                    continue
                # /proc/net/route uses little-endian hex for IPs
                # Default route has Destination=00000000 and a Gateway field
                if fields[1] != "00000000":
                    continue
                gw_hex = fields[2]
                if gw_hex == "00000000":
                    continue
                octets = [int(gw_hex[i : i + 2], 16) for i in (6, 4, 2, 0)]
                return ".".join(str(o) for o in octets)
    except Exception as e:
        logger.debug(f"_probe_default_gateway_ip failed: {e}")
    return None


def _probe_check_gateway() -> str:
    """Return ``"ok"`` if a TCP SYN reaches the default gateway.

    Uses port 53 (DNS) by convention - most home routers run a DNS
    resolver, and if they don't, the TCP RST itself counts as
    reachability proof (we'd see ConnectionRefusedError, not timeout).
    """
    gw = _probe_default_gateway_ip()
    if not gw:
        return "fail"
    try:
        with socket.create_connection((gw, 53), timeout=_PROBE_TCP_TIMEOUT_S):
            return "ok"
    except ConnectionRefusedError:
        # The router answered with TCP RST - it's reachable, just not
        # listening on :53. Counts as a healthy LAN gateway for our
        # purposes.
        return "ok"
    except (TimeoutError, socket.timeout, OSError):
        return "fail"


def _probe_check_dns() -> str:
    """Return ``"ok"`` if a public hostname resolves via the system resolver."""
    socket.setdefaulttimeout(_PROBE_DNS_TIMEOUT_S)
    try:
        socket.gethostbyname(_PROBE_DNS_HOST)
        return "ok"
    except (socket.gaierror, socket.timeout, OSError):
        return "fail"
    finally:
        socket.setdefaulttimeout(None)


def _probe_check_internet() -> str:
    """Return ``"ok"`` if a TCP SYN reaches a public anycast endpoint.

    Plain TCP, no TLS - tolerant to a missing system clock.
    """
    try:
        with socket.create_connection(
            (_PROBE_INTERNET_HOST, _PROBE_INTERNET_PORT),
            timeout=_PROBE_TCP_TIMEOUT_S,
        ):
            return "ok"
    except (TimeoutError, socket.timeout, OSError):
        return "fail"


def _probe_check_daemon() -> str:
    """Return ``"ok"``, ``"loading"`` or ``"fail"`` based on the daemon's status.

    The mobile app uses this to wait gracefully when a fresh-booted
    robot has joined WiFi *before* the daemon's backend has finished
    loading - so we don't sling the user into a conversation that
    immediately 503s.
    """
    try:
        payload = _daemon_request(
            "GET", "/api/daemon/status", timeout=_PROBE_DAEMON_TIMEOUT_S
        )
    except urllib.error.HTTPError as e:
        if e.code in (502, 503):
            return "loading"
        return "fail"
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError):
        return "fail"
    except Exception:
        return "fail"

    if not isinstance(payload, dict):
        return "fail"

    # The daemon exposes a `state` field that flips through
    # ``starting`` -> ``loading`` -> ``running``. Anything other than
    # ``running`` is "loading" from the client's POV.
    state = str(payload.get("state", "")).lower()
    if state in ("running", "ready", "ok"):
        return "ok"
    if state in ("starting", "loading", "booting", "initializing"):
        return "loading"
    # Unknown state shape (older daemons): if we got a 200 with a JSON
    # body, treat the daemon as up.
    return "ok"


def _wifi_probe() -> str:
    """Diagnose end-to-end reachability from the robot's POV.

    Returns a JSON string (single BLE write fits in default MTU):

        {"wlan":"ok","gateway":"ok","dns":"ok","internet":"ok","daemon":"ok"}

    All four network checks run in parallel and are bounded by
    ``_PROBE_TOTAL_TIMEOUT_S`` so the BT command loop never stalls.
    """
    # WLAN status comes from the same helper used for the BLE TLV 0x03
    # network_mode byte - keep them aligned.
    mode_byte = _current_network_mode_byte()
    if mode_byte == ADVERT_NETWORK_MODE_CONNECTED:
        wlan = "ok"
    elif mode_byte == ADVERT_NETWORK_MODE_HOTSPOT:
        wlan = "hotspot"
    else:
        wlan = "fail"

    result: dict[str, str] = {"wlan": wlan}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = {
            "gateway": ex.submit(_probe_check_gateway),
            "dns": ex.submit(_probe_check_dns),
            "internet": ex.submit(_probe_check_internet),
            "daemon": ex.submit(_probe_check_daemon),
        }
        for name, fut in futures.items():
            try:
                result[name] = fut.result(timeout=_PROBE_TOTAL_TIMEOUT_S)
            except concurrent.futures.TimeoutError:
                result[name] = "fail"
            except Exception as e:
                logger.debug(f"_wifi_probe[{name}] failed: {e}")
                result[name] = "fail"

    return json.dumps(result, separators=(",", ":"))


def _hex_id_prefix_bytes(value: str, prefix_bytes: int) -> bytes:
    """Decode the first ``prefix_bytes`` of a hex string ``value``.

    Tolerant by design: dashes (eg from a UUID like
    ``204c2579-28c9-4d22-...``) are stripped, the input is lowercased,
    and any decoding failure yields ``b""`` so callers can simply skip
    the corresponding TLV instead of crashing the daemon.
    """
    if not value:
        return b""
    candidate = value.strip().lower().replace("-", "")
    needed_hex_chars = prefix_bytes * 2
    if len(candidate) < needed_hex_chars:
        return b""
    try:
        return bytes.fromhex(candidate[:needed_hex_chars])
    except ValueError:
        return b""


def _install_id_prefix_bytes(install_id: str) -> bytes:
    """Return the first ``ADVERT_INSTALL_ID_PREFIX_BYTES`` bytes of ``install_id``.

    ``install_id`` is the daemon's UUID4 hex (32 chars). We take its
    first 16 hex chars and decode them to 8 raw bytes (= 64 bits of
    uniqueness, collision-safe for any realistic fleet size).
    """
    return _hex_id_prefix_bytes(install_id, ADVERT_INSTALL_ID_PREFIX_BYTES)


def _central_peer_id_prefix_bytes(central_peer_id: str) -> bytes:
    """Return the first ``ADVERT_CENTRAL_PEER_ID_PREFIX_BYTES`` of the central peerId.

    Central hands us back a UUID-shaped peerId (with dashes) on the
    welcome frame. We compress it the same way as install_id so the
    mobile parser can match it against the central listing's peerId
    truncated to its first 16 hex chars.
    """
    return _hex_id_prefix_bytes(central_peer_id, ADVERT_CENTRAL_PEER_ID_PREFIX_BYTES)


def _current_network_mode_byte() -> int | None:
    """Derive the BLE TLV 0x03 byte from the local network state.

    Reads ``ip -4 addr show`` once, classifies the wlan0 IP literal
    against the well-known hotspot range (``10.42.0.1``) and returns:

        - ``0x02 CONNECTED`` if any real interface has an IP that
          isn't the hotspot literal,
        - ``0x01 HOTSPOT``   if wlan0 holds the hotspot fallback
          address (and no other real interface is up),
        - ``0x00 OFFLINE``   if no IPv4 is configured anywhere.

    Returns ``None`` only if the subprocess itself fails (rare); the
    caller then drops the TLV rather than emitting a misleading
    OFFLINE byte that could confuse mobile clients during boot.
    """
    try:
        result = subprocess.run(
            ["ip", "-4", "addr", "show"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception as e:
        logger.debug(f"_current_network_mode_byte: subprocess failed: {e}")
        return None

    interfaces: dict[str, str] = {}
    current_iface = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line and not line.startswith("inet"):
            parts = line.split(":")
            if len(parts) >= 2 and parts[1].strip():
                iface = parts[1].strip()
                if iface != "lo":
                    current_iface = iface
        elif line.startswith("inet ") and current_iface:
            inet_parts = line.split()
            if len(inet_parts) >= 2:
                ip_with_mask = inet_parts[1]
                ip_addr = ip_with_mask.split("/")[0]
                interfaces[current_iface] = ip_addr

    if not interfaces:
        return ADVERT_NETWORK_MODE_OFFLINE

    # Connected if ANY interface holds a non-hotspot IP. Wired
    # tether (usb0, eth0, ...) counts as connected even when wlan0
    # is in hotspot fallback - the robot can still talk to central
    # via the tether, and the user has nothing to set up Wi-Fi-wise.
    has_connected_iface = any(
        not ip.startswith("10.42.0.1") for ip in interfaces.values()
    )
    if has_connected_iface:
        return ADVERT_NETWORK_MODE_CONNECTED

    # All interfaces are on the hotspot literal ⇒ daemon is in setup
    # mode (broadcasting its own AP).
    return ADVERT_NETWORK_MODE_HOTSPOT


def encode_advert_manufacturer_data(
    install_id: str,
    central_peer_id: str = "",
    network_mode: int | None = None,
) -> dict:
    """Build the BLE advertisement ``ManufacturerData`` payload.

    Returns a ``{manufacturer_id: dbus.Array[byte]}`` dict suitable
    for the BlueZ ``ManufacturerData`` advertisement property, with a
    versioned TLV stream carrying the daemon's install_id prefix,
    (when available) its central peerId prefix, and (when known) a
    1-byte network_mode descriptor. See the ``ADVERT_*`` constants
    at the top of this module for the on-wire layout.

    ``network_mode`` should be one of ``ADVERT_NETWORK_MODE_*`` or
    ``None`` (TLV omitted - signals "old daemon" to mobile clients).

    Returns an empty dict when no payload at all is available yet so
    the advert just goes out without a ManufacturerData section
    (rather than a malformed one).
    """
    install_prefix = _install_id_prefix_bytes(install_id)
    cp_prefix = _central_peer_id_prefix_bytes(central_peer_id)
    if not install_prefix and not cp_prefix and network_mode is None:
        return {}

    payload = bytearray()
    payload.append(ADVERT_FORMAT_VERSION)
    if install_prefix:
        payload.append(ADVERT_TLV_INSTALL_ID)
        payload.append(len(install_prefix))
        payload.extend(install_prefix)
    if cp_prefix:
        payload.append(ADVERT_TLV_CENTRAL_PEER_ID)
        payload.append(len(cp_prefix))
        payload.extend(cp_prefix)
    if network_mode is not None:
        payload.append(ADVERT_TLV_NETWORK_MODE)
        payload.append(1)
        payload.append(network_mode & 0xFF)

    return {
        dbus.UInt16(POLLEN_MANUFACTURER_ID): dbus.Array(
            [dbus.Byte(b) for b in payload], signature="y"
        )
    }


def get_pin() -> str:
    """Extract the last 5 digits of the serial number from dfu-util -l output."""
    default_pin = "46879"
    try:
        result = subprocess.run(["dfu-util", "-l"], capture_output=True, text=True)
        lines = result.stdout.splitlines()
        for line in lines:
            if "serial=" in line:
                # Extract serial number
                serial_part = line.split("serial=")[-1].strip().strip('"')
                if len(serial_part) >= 5:
                    return serial_part[-5:]
        return default_pin  # fallback if not found
    except Exception as e:
        logger.error(f"Error getting pin from serial: {e}")
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
# Main
# =======================
def main():
    """Run the Bluetooth Command Service."""
    pin = get_pin()

    bt_service = BluetoothCommandService(device_name="ReachyMini", pin_code=pin)
    bt_service.run()


if __name__ == "__main__":
    main()
