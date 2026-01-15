#!/usr/bin/env python3
"""Bluetooth service for Reachy Mini using direct DBus API.

Includes a fixed NoInputNoOutput agent for automatic Just Works pairing.
"""
# mypy: ignore-errors

import logging
import os
import subprocess
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
        logger.info(f"Command received: {response}")


class ResponseCharacteristic(Characteristic):
    """Response Characteristic."""

    def __init__(self, bus, index, service):
        """Initialize the Response Characteristic."""
        super().__init__(bus, index, RESPONSE_CHAR_UUID, ["read", "notify"], service)


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

    def _handle_command(self, value: bytes) -> str:
        command_str = value.decode("utf-8").strip()
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
        elif command_str.startswith("PIN_"):
            pin = command_str[4:].strip()
            if pin == self.pin_code:
                self.connected = True
                return "OK: Connected"
            else:
                return "ERROR: Incorrect PIN"

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
        service_manager.RegisterApplication(
            self.app.get_path(),
            {},
            reply_handler=lambda: logger.info("GATT app registered"),
            error_handler=lambda e: logger.error(f"Failed to register GATT app: {e}"),
        )

        # Register advertisement
        ad_manager = dbus.Interface(adapter, LE_ADVERTISING_MANAGER_IFACE)
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

        # Setup periodic network status updates (every 10 seconds)
        GLib.timeout_add_seconds(10, self.app.reachy_status.update_network_status)

        logger.info(f"✓ Bluetooth service started as '{self.device_name}'")

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
            self.mainloop.quit()


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
