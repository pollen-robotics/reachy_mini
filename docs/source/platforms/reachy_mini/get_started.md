# Reachy Mini (Wireless) - Setup Guide

The **Reachy Mini (Wireless)** is the autonomous version powered by a Raspberry Pi Compute Module 4 (CM4). It runs independently using its internal battery and Wi-Fi connection.

## 1. 🔧 Assembly

Reachy Mini comes as a kit. Building it is the first step of your journey!

* **Time required:** 2 to 3 hours.
* **Tools:** Everything is included in the box.
* **Instructions:** We strongly recommend following the video guide alongside the manual.

> **💡 Pro Tip:** We strongly recommend having the **Online Guide** or the **assembly video** open alongside the paper booklet (see below). The online version includes short video extracts for every step, which makes the assembly much easier to understand.

| **📖 Interactive Digital Guide** | **📺 Full Assembly Video** |
| :---: | :---: |
| [![Assembly Guide](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/digital_assembly_guide_preview_mini.jpg)](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini_Assembly_Guide)<br>[**Open Step-by-Step Guide**](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini_Assembly_Guide)<br>*(Includes short video loops)* | [![Watch on YouTube](https://img.youtube.com/vi/WeKKdnuXca4/maxresdefault.jpg)](https://www.youtube.com/watch?v=WeKKdnuXca4)<br>[**Watch on YouTube**](https://www.youtube.com/watch?v=WeKKdnuXca4)<br>*(Video with sections for each step)* |


## 2. 🛜 First Boot & Wi-Fi Configuration

Once assembled, you need to connect the robot to your Wi-Fi network and link it to your Hugging Face account. The easiest way is the **Reachy Mini app** on your phone.

1. **Power On:** Turn on your Reachy Mini.
2. **Get the app:** Install **Reachy Mini** from the [App Store](https://apps.apple.com/us/app/reachy-mini/id6766823749) or [Google Play](https://play.google.com/store/apps/details?id=com.pollenrobotics.reachymini), and sign in with your [Hugging Face](https://huggingface.co) account.
3. **Set it up:** Tap **Set up a new Reachy** and follow the steps. The app finds the robot over Bluetooth, connects it to your Wi-Fi, updates its software if needed and links it to your account.
![Set up a new Reachy](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/mobile_app/first-reachy.png)

👉 Step by step, with screenshots: **[Mobile app: getting started](../mobile_app/get_started.md)**.

> **Pick a network where devices can see each other.** Guest, hotel and conference networks often block it (client isolation). See [Bluetooth tools](../mobile_app/bluetooth_tools.md) if the robot gets stuck on such a network.

<details>
<summary><strong>No phone? Use Reachy Mini Control on your computer</strong></summary>

1. **Download Reachy Mini Control** from the [official website](https://pollen-robotics.com/reachy-mini/download/).
2. **Run the App:** Open **Reachy Mini Control** and click on the bottom link **"First time connecting..."**.
![First time connecting](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/first-setup-wifi-1.png)
3. **Follow the Instructions:** The app will guide you through the connection process. It will ask you to connect to the robot's Wi-Fi AP and then configure your Wi-Fi.
![Configuring your WiFi](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/first-setup-wifi-2.png)

</details>

## 3. 🔄 Update System

Before going further, it is highly recommended to update your robot to the latest version.

**With the mobile app**, there's nothing to do: it updates the robot during setup, and asks you to update when you connect to a robot with older software. You can also update over Bluetooth from the [Bluetooth tools](../mobile_app/bluetooth_tools.md) (**Update software**).

**With Reachy Mini Control:**

1. Connect to your robot using **Reachy Mini Control**.
2. Once connected, click on the **"⚙️"** settings tab.
![Settings Tab](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/control-app-settings.png)
3. Go to the **System Updates** section.
![System Update Section](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/control-app-update.png)
4.  **Install:** If a new version is available, follow the on-screen instructions to install it.


## 4. 🕹️ Next Step: Using the Robot

Now that your robot is online and up to date, you can start controlling it!

👉 **[Use it from your phone](../mobile_app/using.md)**: talk with it, run apps, drive it remotely.

👉 **[Go to the Usage Guide](usage.md)** to learn how to:
* Use the **Reachy Mini Control**.
* Install and run **Apps** (like Conversation or Games).
* Program your Reachy with **Python**.

## 5. 💻 Advanced: Connect directly to the internal Raspberry Pi via SSH

If you need to connect to Reachy Mini's internal Raspberry Pi via SSH, credentials are:

```
username: pollen
password: root
```

Once connected via SSH, you can check the integrity of your Raspberry Pi and robot setup with:

```bash
reachyminios_check
```

## ❓ Troubleshooting

Encountering an issue? 👉 **[Check the Troubleshooting & FAQ Guide](../../troubleshooting.md)**

## Expert Mode

If you need to reinstall the Raspberry Pi from scratch or create a custom image, follow the expert guides.

**[Reflash the ISO](reflash_the_rpi_ISO.md)**

**[Install Daemon from a Specific Branch](install_daemon_from_branch.md)**

**[Development Workflow](development_workflow.md)** - Best practices for developing and testing code on the Wireless Reachy Mini
