# Reachy Mini mobile app: getting started

The Reachy Mini app lets you set up and use your robot from your phone (iOS and Android): connect it to Wi-Fi, talk with it, run apps from the Hugging Face store and drive its head remotely.

Download it from the [App Store](https://apps.apple.com/us/app/reachy-mini/id6766823749) or [Google Play](https://play.google.com/store/apps/details?id=com.pollenrobotics.reachymini).

You need a [Hugging Face](https://huggingface.co) account. Your robots are linked to it, and the app finds them through it, at home or away.

> **Reachy Mini Lite:** no setup in the mobile app. It shows up through [Reachy Mini Control](https://github.com/pollen-robotics/reachy-mini-desktop-app/releases/latest) running on its computer. See [Using Reachy Mini Lite: from your phone](../reachy_mini_lite/usage.md#3-from-your-phone).

## 1. Sign in

On first launch, the app explains what it does with your data. Tap **Accept and continue**, then **Continue with Hugging Face** and sign in.

![Consent screen](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/consent.png)
![Sign in](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/sign-in.png)

## 2. Your Reachies

**Your Reachies** lists every robot linked to your account and says which ones are online. Tap one to connect. A robot already used by someone else shows **In use**.

![Robot list](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/robot-list.png)

At the top right, the wrench opens the [Bluetooth tools](bluetooth_tools.md) and **?** opens **Help & support** (see [Using the app](using.md)). The icon next to your name signs you out.

## 3. Set up a new Reachy (Wireless)

With no robot online yet, the app shows **Meet Your First Reachy Mini**. Otherwise, the button is at the bottom of **Your Reachies**.

![Meet your first Reachy Mini](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/first-reachy.png)

Power the robot on and tap **Set up a new Reachy**. Stay close to it: setup starts over Bluetooth, so the robot doesn't need to be on Wi-Fi yet. There are 4 steps.

1. **Find me**: the app scans for robots nearby. Tap yours.

   ![Find me](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/setup-find.png)

   Then **Unlock Me**: enter the PIN, the last 5 digits of the serial number (SN) on the label under the robot's base, and tap **Verify**.

   ![Unlock me](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/setup-pin.png)

2. **Get Me Online**: pick your Wi-Fi network, type its password and tap **Connect**. If the robot is already online, you can keep its network or tap **Use a different network**.

   ![Get me online](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/setup-wifi.png)

   > Pick a network where devices can reach each other. Guest, hotel and conference networks often block this (client isolation) and setup fails at the last step. If that happens, see [Bluetooth tools](bluetooth_tools.md) to make the robot forget that network.

3. **Bringing it online**: the robot joins the network. If its software is too old, the app updates it first (**Getting the latest software**).

4. **Link me to Hugging Face**: keep your phone on the same Wi-Fi as the robot. Tap **Sign in with Hugging Face**, then **Copy code & open Hugging Face** and confirm the code in the browser.

   ![Link to Hugging Face](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/setup-hf-link.png)

Once the robot is registered (**Almost there**), the app connects to it.

### If setup gets stuck

- **Allow Bluetooth access**: the app needs Bluetooth permission. On iOS, tap **Open Settings**, allow it, then **Try again**.
- **Forget the old pairing**: the phone remembers an old pairing with the robot. Remove the robot from your phone's Bluetooth settings, then try again.
- **Reflash required**: the robot software is too old to update over the air. Use the [Reachy Mini Flasher](../reachy_mini/reflash_the_rpi_ISO.md).

## 4. First wake-up

The first time you connect to a robot, a short guide checks its motors, camera, microphone and speaker. See [First wake-up](first_wake_up.md).

Then: [Using the app](using.md).
