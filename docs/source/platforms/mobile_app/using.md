# Using the mobile app

Tap a robot in **Your Reachies** to connect. The app wakes the robot up and opens three tabs: **Conversation**, **Apps** and **Telepresence**.

At the top, next to the robot's name, you see how it is connected and the link latency. The gear icon opens the [settings](#settings), the power icon ends the session and puts the robot to sleep.

If the robot software is too old, the app asks you to update it first (**Update now**).

## Conversation

Tap the orb to start talking with your Reachy. While you talk, side buttons let you mute your microphone or end the conversation.

![Conversation](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/conversation.png)

Tap **Personality** to change how Reachy talks: pick a character or an assistant, or tap **Create your own personality** and describe one. Creating a personality uses Hugging Face inference, so your account needs an inference provider enabled.

![Personalities](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/personality.png)

## Apps

Reachy Mini apps are published by the community on Hugging Face. There are two kinds:

- **Phone apps** (JavaScript, **Runs on phone**): they open full screen in the mobile app and drive the robot remotely. They stop when you close them.
- **Robot apps** (Python, **Runs on Reachy**): they are installed on the robot and run there, even when your phone is away.

The Apps tab shows **On your robot** (the robot apps installed on Reachy) and **Your launcher** (your pinned phone apps).

![Apps](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/apps.png)

### Phone apps

Tap **Browse the store**. On the **On your phone** tab, tap **Try** to open an app, or **Pin** to add it to your launcher. **Edit** on the launcher removes pinned apps.

![Store](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/store.png)

### Robot apps

Tap **Browse robot apps** (or the **On Reachy** tab of the store), then **Get** on an app.

![Robot apps in the store](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/store-robot.png)

1. Tap **Install on Reachy**. The robot downloads the app and installs it. **Show full log** shows the details. It can take a few minutes.

   ![Robot app details](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/robot-app-sheet.png)
   ![Installing a robot app](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/robot-app-installing.png)

2. Tap **Start**. The app shows **Running**, both in its details and under **On your robot**. **Stop** (or the square button in the list) stops it, **Restart** restarts it.

   ![Running robot app](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/robot-app-running.png)

3. Once stopped, the details offer **Update** to get the latest version and **Uninstall** to remove the app from the robot.

   ![Uninstall a robot app](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/robot-app-uninstall.png)

Under any app, **...** lets you report it, hide its author's apps, or open it on Hugging Face.

### Developers

Want to build your own app and see it in the store?

- **Phone app** (JavaScript, runs in the mobile app): see [JavaScript SDK & Web Apps](../../SDK/javascript-sdk.md).
- **Robot app** (Python, runs on Reachy): see [Building & Publishing Apps](../../SDK/apps.md).

## Telepresence

See through the robot's camera and move its head with the **Head** joystick.

![Telepresence](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/telepresence.png)

The buttons around the joystick:

- **Microphone**: mute or unmute your voice. When on, you talk through the robot's speaker.
- **Speaker**: mute or unmute what the robot hears, played on your phone.
- **Smiley**: opens the wheel of emotions. Tap one to make the robot play it.
- **Frame**: recenters the robot's head.

The gear icon sets the head roll, body rotation, antennas, video latency and robot audio. The arrow leaves telepresence.

## Settings

The gear icon at the top of a session opens the settings.

### Audio and conversation

![Settings](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/settings.png)

- **Audio**: speaker and microphone volume.
- **Language**: the language Reachy speaks and listens in.
- **Privacy & memory**:
  - **Let Reachy see**: Reachy takes a camera snapshot when you ask it to look at something.
  - **Long-term memory**: Reachy remembers a few facts you share, across conversations. **Clear memory** deletes them.
- **Without the phone**: **Talk when the antennas are touched** wakes Reachy up and starts a conversation when you touch its antennas while it sleeps.

These settings are locked while you talk: stop the conversation to change them.

### Robot

![Robot settings](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/settings-robot.png)

- **Name**: type a new name to rename the robot. It updates right away in your robot list, no restart needed.
- **Wi-Fi**: **Change Wi-Fi network** moves the robot to another network, or updates its Wi-Fi password. It runs over Bluetooth, so stay close to the robot. See [Bluetooth tools](bluetooth_tools.md).
- **Account**: **Sign out this Reachy** unlinks the robot from your Hugging Face account. You'll need to [set it up again](get_started.md) to use it.

### About & diagnostics

![About and diagnostics](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/about.png)

Useful when something goes wrong, or when you ask for support:

- **Robot**: the robot's identifier.
- **Connection**: how the phone reaches the robot (**LAN** on the same network, or remotely through Hugging Face), throughput, latency, the robot's IP and recent connection incidents.
- **Software**: the robot software (**Daemon**) and mobile app versions.
- **Privacy**: **Share anonymous usage data** is on by default. Turn it off to stop sending anonymous usage statistics. See [Telemetry](../../SDK/telemetry.md) for what is sent.
- **Account**: the Hugging Face account you're signed in with.
- **Logs**: the robot software's live log. The copy button copies it, to paste in a bug report.

## Help & support

On **Your Reachies**, the **?** icon opens Help & support: light or dark theme, support contact, the troubleshooting guide, this documentation, Discord, GitHub and the legal pages. The app version is at the bottom, include it when you ask for support.

![Help and support](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/help.png)
