# Bluetooth reset and restart tools (Wireless)

If you need to reset the robot's Wi-Fi hotspot, restart the daemon, or run other maintenance commands over Bluetooth, use one of the options below. **Prefer the official Reachy Mini Control app** when you are on a laptop or desktop: it bundles a Bluetooth console and avoids juggling a separate browser or phone app.

## 1. Reachy Mini Control — Bluetooth console (recommended)

Use the **Reachy Mini Control** desktop app ([download](https://hf.co/reachy-mini/#/download)). Bluetooth must be enabled on your computer.

1. Open **Reachy Mini Control**. On the **Connect to Reachy** screen (before you start a session), scroll to the footer and click **First time WiFi setup**.

[![control-app-first-time-wifi-setup.png](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/control-app-first-time-wifi-setup.png)]()

2. In the **First Time WiFi Setup** wizard, use the footer link **Try the Bluetooth Console** (for example if the hotspot is not detected, or whenever you want the console directly).

[![control-app-wifi-wizard-bluetooth-console.png](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/control-app-wifi-wizard-bluetooth-console.png)]()

From the Bluetooth console you can reset the hotspot, check network status, and run the same kind of commands as in the other methods below.

> **Note:** In some app versions the entry link may read **First time connecting to your WiFi...** instead of **First time WiFi setup**; it opens the same Wi-Fi setup flow.

## 2. Web Bluetooth dashboard (phone or desktop browser)

With a smartphone or desktop and a **Bluetooth-capable browser** (Chrome, Opera, or Edge), open the standalone tool: [Bluetooth tool](https://pollen-robotics.github.io/reachy_mini/). You can reset the hotspot, check network status, and more without installing the desktop app.

If your device or browser does not support the Web Bluetooth API, use the desktop app (above) or the nRF Connect method below.

## 3. nRF Connect (advanced)

Reachy Mini wireless exposes a Bluetooth GATT interface so you can reset the Wi-Fi hotspot or the daemon from a generic BLE client.

### 3.1 Install nRF Connect

- **Android**: [Google Play](https://play.google.com/store/apps/details?id=no.nordicsemi.android.mcp&hl=en-US&pli=1)
- **iOS**: [App Store](https://apps.apple.com/us/app/nrf-connect-for-mobile/id1054362403)

### 3.2 Scan and connect

1. Open nRF Connect.
2. Scan for devices and select **ReachyMini**.

[![bluetooth_1.jpg](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/bluetooth_1.jpg)]()

3. Connect to the device.

### 3.3 Unknown service & WRITE section

- Open the **Unknown Service**.
- Use the **WRITE** section to send commands.
- You can use the **READ** section above to inspect the robot's responses.

[![bluetooth_2.jpg](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/bluetooth_2.jpg)]()

### 3.4 Sending commands

Commands are sent as **hexadecimal strings** (or as plain text such as `STATUS` or `CMD_HOTSPOT`). Use [this converter](https://www.rapidtables.com/convert/number/ascii-to-hex.html) if you need ASCII → hex.

**Always send the PIN before any other command.**

#### PIN code

The PIN is the **last 5 digits** of the robot's serial number. For example, if the serial number is `xxxxxxxx4918400018`, the PIN is `00018` — send `PIN_00018` (or the hex value below).

#### Command reference

| Command                | Hex value (after `0x`)              |
|------------------------|-------------------------------------|
| STATUS                 | 535441545553                        |
| PIN_00018              | 50494E5F3030303138                  |
| CMD_HOTSPOT            | 434D445F484F5453504F54              |
| CMD_RESTART_DAEMON     | 434D445F524553544152545F4441454D4F4E |
| CMD_SOFTWARE_RESET     | 434D445F534F4654574152455F5245534554 |

If you use **CMD_SOFTWARE_RESET**, the robot reboots and may take **about 5 minutes** before it is reachable again.

#### Tips

It is convenient to save frequently used commands for reuse.

[![bluetooth_3.jpg](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/bluetooth_3.jpg)]()
[![bluetooth_4.jpg](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/bluetooth_4.jpg)]()
