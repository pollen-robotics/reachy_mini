# Bluetooth tools

The Bluetooth tools let you check, fix and update a Reachy Mini Wireless that isn't reachable over the network. They only need you to be close to the robot.

Open them from the wrench icon at the top of **Your Reachies**, or from **Settings › Wi-Fi › Change Wi-Fi network** during a session. Tap **Start**, pick your robot in **Nearby Reachies**, then enter the 5-digit code printed under its base and tap **Continue**.

![Bluetooth tools](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/ble-menu.png)

## Robot

- **Status**: robot software version, network mode, Wi-Fi network, IP address, saved networks and hardware ID.
- **Daemon logs**: live logs of the robot software, useful when asking for support.
- **Update software**: installs the latest robot software.

## Network

- **Change Wi-Fi network**: connect the robot to another network.

## Recovery

| Entry | What it does |
|-------|--------------|
| **Restart the robot software** | Restarts the daemon, not the whole robot. |
| **Switch to hotspot mode** | Leaves the current Wi-Fi and broadcasts the robot's own `reachy-mini-ap` network. The robot rejoins its saved networks at the next boot. |
| **Forget all Wi-Fi networks** | Deletes every saved network. The robot falls back to hotspot mode, then use **Change Wi-Fi network** or set it up again. |
| **Restore factory software** | Reinstalls the robot software from its recovery copy. Only if the robot no longer starts after an update. It can take a few minutes. |

Each entry asks for confirmation first. They all restart the robot software, which drops the Bluetooth link: that's expected.

### The robot joined a network it can't be reached on

This happens when setup used a guest, hotel or conference network that blocks devices from talking to each other: the robot keeps rejoining that network at every boot. Use **Forget all Wi-Fi networks**, then set the robot up again on another network.

![Forget all Wi-Fi networks](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/ble-forget-confirm.png)

Without the mobile app, the same commands are available from the [desktop app and other Bluetooth tools](../reachy_mini/reset.md).
