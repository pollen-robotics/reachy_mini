#!/usr/bin/env python3
"""
BLE-only test client that forces Low Energy connections.

This client explicitly uses BLE scanning and connects only via GATT/LE.
"""

import asyncio
import logging
import sys

from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# UUIDs from the service
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
COMMAND_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef1"
RESPONSE_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef2"


async def scan_for_ble_device(
    device_name_or_address: str, duration: float = 15.0, max_attempts: int = 5
) -> BLEDevice:
    """
    Scan specifically for BLE devices only with multiple retry attempts.
    This uses callback-based scanning which is more reliable.
    """

    for attempt in range(max_attempts):
        if attempt > 0:
            print(f"\n🔄 Retry attempt {attempt + 1}/{max_attempts}")
            await asyncio.sleep(2)  # Wait between attempts

        print(
            f"🔍 Scanning for BLE device '{device_name_or_address}' (up to {duration}s)..."
        )

        found_device = None
        all_devices = {}
        scan_count = [0]  # Use list to modify in callback

        def detection_callback(
            device: BLEDevice, advertisement_data: AdvertisementData
        ):
            """Callback for each detected device."""
            nonlocal found_device

            scan_count[0] += 1

            # Store all devices
            all_devices[device.address] = (device, advertisement_data)

            # Check if this matches our target
            name = device.name or advertisement_data.local_name or ""
            address_match = device.address.upper().replace(
                ":", ""
            ) == device_name_or_address.upper().replace(":", "")
            name_match = device_name_or_address.lower() in name.lower()

            if address_match or name_match:
                found_device = device
                print(
                    f"✅ Found: {name} ({device.address}) - RSSI: {getattr(advertisement_data, 'rssi', 'N/A')}"
                )

        # Create BLE scanner with callback
        scanner = BleakScanner(
            detection_callback=detection_callback,
            scanning_mode="active",  # Active scanning for more data
        )

        # Start scanning
        try:
            await scanner.start()
        except Exception as e:
            logger.warning(f"Failed to start scanner: {e}")
            continue

        # Scan for specified duration
        start_time = asyncio.get_event_loop().time()
        last_report = 0

        while (asyncio.get_event_loop().time() - start_time) < duration:
            await asyncio.sleep(0.5)

            elapsed = asyncio.get_event_loop().time() - start_time

            # If found device, wait a bit more for stable signal then exit
            if found_device:
                if elapsed > 2:  # Wait at least 2 seconds after finding
                    break

            # Progress indicator every 2 seconds
            if int(elapsed) >= last_report + 2:
                last_report = int(elapsed)
                print(
                    f"  {int(elapsed)}s: {len(all_devices)} devices, {scan_count[0]} advertisements"
                )

        try:
            await scanner.stop()
        except Exception as e:
            logger.warning(f"Error stopping scanner: {e}")

        if found_device:
            return found_device

        # Not found in this attempt
        print(f"  ❌ Not found in attempt {attempt + 1}")
        if all_devices:
            print(f"  Found {len(all_devices)} other BLE devices:")
            for addr, (dev, adv) in list(all_devices.items())[:5]:
                name = dev.name or adv.local_name or "Unknown"
                rssi = getattr(adv, "rssi", "N/A")
                print(f"    • {name} - {addr} (RSSI: {rssi})")

    # After all attempts
    print(
        f"\n❌ BLE device '{device_name_or_address}' not found after {max_attempts} attempts!"
    )
    return None


async def send_command_ble(command: str, device_id: str = "ReachyMini"):
    """Send a command via BLE GATT."""

    print("=" * 60)
    print("  BLE-Only Bluetooth Client")
    print("=" * 60)
    print(f"Command: {command}")
    print(f"Target: {device_id}")
    print("=" * 60)
    print()

    # Scan for BLE device with retries
    device = await scan_for_ble_device(device_id, duration=30.0, max_attempts=3)
    if not device:
        print("\n💡 Make sure:")
        print("  • The BLE server is running")
        print("  • You're using the correct device name or MAC address")
        print("  • The device is within range")
        return 1

    print(
        f"\n📡 Connecting to {device.name or 'Unknown'} ({device.address}) via BLE..."
    )

    try:
        # Connect via BLE GATT
        # Using the device from BLE scan should force LE connection
        async with BleakClient(device, timeout=20.0) as client:
            print(f"✅ Connected via BLE!")

            # Discover services
            print(f"🔍 Discovering GATT services...")

            # Check if our service exists
            services = client.services
            our_service = None
            for service in services:
                if SERVICE_UUID.lower() in service.uuid.lower():
                    our_service = service
                    print(f"✓ Found command service")
                    break

            if not our_service:
                print(f"⚠️  Service {SERVICE_UUID} not found!")
                print(f"Available services:")
                for svc in list(services)[:5]:
                    print(f"  • {svc.uuid}")
                return 1

            # Find characteristics
            command_char = None
            response_char = None

            for char in our_service.characteristics:
                if COMMAND_CHAR_UUID.lower() in char.uuid.lower():
                    command_char = char
                    print(f"✓ Found command characteristic")
                elif RESPONSE_CHAR_UUID.lower() in char.uuid.lower():
                    response_char = char
                    print(f"✓ Found response characteristic")

            if not command_char or not response_char:
                print(f"❌ Required characteristics not found!")
                return 1

            # Send command
            print(f"\n📤 Sending: '{command}'")
            await client.write_gatt_char(
                command_char, command.encode("utf-8"), response=False
            )
            print(f"✓ Command sent")

            # Wait for processing
            print(f"⏳ Waiting for response...")
            await asyncio.sleep(0.8)

            # Read response
            response_bytes = await client.read_gatt_char(response_char)
            response = response_bytes.decode("utf-8")

            print(f"\n📥 Response: '{response}'")
            print(f"\n✅ Success!")
            return 0

    except Exception as e:
        error_str = str(e)

        if "br-connection-profile-unavailable" in error_str.lower():
            print(f"\n❌ ERROR: Tried to connect via Bluetooth Classic instead of BLE!")
            print(f"\nThis means BlueZ is confused about the device type.")
            print(f"\nTry these solutions:")
            print(f"  1. Remove the device from BlueZ cache:")
            print(f"     bluetoothctl remove {device.address}")
            print(f"  2. Restart Bluetooth:")
            print(f"     sudo systemctl restart bluetooth")
            print(f"  3. Disable BR/EDR on the server (see docs)")
            return 1
        else:
            print(f"❌ Error: {e}")
            logger.exception("Connection failed:")
            return 1


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(
            "Usage: python test_bluetooth_client_ble_only.py <command> [device_name_or_mac]"
        )
        print("\nExamples:")
        print('  python test_bluetooth_client_ble_only.py "PING"')
        print('  python test_bluetooth_client_ble_only.py "PING" ReachyMini')
        print('  python test_bluetooth_client_ble_only.py "STATUS" 88:A2:9E:3B:88:C6')
        sys.exit(1)

    # Parse arguments
    device_id = "ReachyMini"
    command_parts = sys.argv[1:]

    # Check if last argument looks like a device identifier
    if len(command_parts) > 1:
        last_arg = command_parts[-1]
        # MAC address or device name (starts with capital)
        if ":" in last_arg or (last_arg[0].isupper() and len(last_arg) > 3):
            device_id = command_parts[-1]
            command_parts = command_parts[:-1]

    command = " ".join(command_parts)

    exit_code = asyncio.run(send_command_ble(command, device_id))
    sys.exit(exit_code)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(1)
