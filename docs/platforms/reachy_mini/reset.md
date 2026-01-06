# Smartphone simple bluetooth dashboard

With a smarpthone and a Bluetooth Web API able browser (chrome based/opera/edge) you simply go there 👉 [Bluetooth tool](https://pollen-robotics.github.io/reachy_mini/) and reset hotspot, check network status or other tasks.

If your device/browser isn't compatible, please check the other solution.


# ReachyMini nRF Connect Guide

Reachy Mini wireless emits a Bluetooth signal that allows you to reset the Wi-Fi hotspot or the daemon.

## 1. Install nRF Connect
- **Android**: [Download here](https://play.google.com/store/apps/details?id=no.nordicsemi.android.mcp&hl=en-US&pli=1)
- **iOS**: [Download here](https://apps.apple.com/us/app/nrf-connect-for-mobile/id1054362403)


## 2. Scan and Connect
1. Open nRF Connect.
2. Scan for devices and select **ReachyMini**.
[![bluetooth_1.jpg](/docs/assets/bluetooth_1.jpg)]()
3. Connect to the device.



## 3. Unknown Service & WRITE Section
- Navigate to the **Unknown Service**.
- Locate the **WRITE** section.
[![bluetooth_2.jpg](/docs/assets/bluetooth_2.jpg)]()



## 4. Sending Commands
Commands are sent as **hexadecimal strings**. Use [this tool](https://www.rapidtables.com/convert/number/ascii-to-hex.html) to convert ASCII to hex if needed.


### Available Commands
| Command                | Hex Value (send after `0x`)       |
|------------------------|-----------------------------------|
| STATUS                 | 535441545553                      |
| PIN_00018              | 50494E5F3030303138                |
| CMD_HOTSPOT            | 434D445F484F5453504F54            |
| CMD_RESTART_DAEMON     | 434D445F524553544152545F4441454D4F4E |
| CMD_SOFTWARE_RESET     | 434D445F534F4654574152455F5245534554 |


### PIN Code

The PIN code here is 00018, for example. The PIN is the **last 5 digits** of the robot's serial number.


### Tips

It is good practice to save the commands for later use.
[![bluetooth_3.jpg](/docs/assets/bluetooth_3.jpg)]()
[![bluetooth_4.jpg](/docs/assets/bluetooth_4.jpg)]()




