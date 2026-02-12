# Reachy Mini (Wireless) - Setup Guide

The **Reachy Mini (Wireless)** is the autonomous version powered by a Raspberry Pi 4. It runs independently using its internal battery and Wi-Fi connection.

## 1. 🔧 Assembly

Reachy Mini comes as a kit. Building it is the first step of your journey!

* **Time required:** 2 to 3 hours.
* **Tools:** Everything is included in the box.
* **Instructions:** We strongly recommend following the video guide alongside the manual.

> **💡 Pro Tip:** We strongly recommend having the **Online Guide** or the **assembly video** open alongside the paper booklet (see below). The online version includes short video extracts for every step, which makes the assembly much easier to understand.

| **📖 Interactive Digital Guide** | **📺 Full Assembly Video** |
| :---: | :---: |
| [![Assembly Guide](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/digital_assembly_guide_preview_mini.jpg)](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini_Assembly_Guide)<br>[**Open Step-by-Step Guide**](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini_Assembly_Guide)<br>*(Includes short video loops)* | [![Watch on YouTube](https://img.youtube.com/vi/WeKKdnuXca4/maxresdefault.jpg)](https://www.youtube.com/watch?v=WeKKdnuXca4)<br>[**Watch on YouTube**](https://www.youtube.com/watch?v=WeKKdnuXca4)<br>*(Video with sections for each step)* |


## 2. 🛜 First Boot & Wi-Fi Configuration

Once assembled, you need to connect the robot to your Wi-Fi network.

1.  **Power On:** Turn on your Reachy Mini.
2.  **Connect to Reachy:** Wait a few moments. The robot will create a Wi-Fi network named **`reachy-mini-ap`**.
    * **Password:** `reachy-mini`
    * *Or scan the QR Code:*
    
    ![QR-Code reachy-mini-ap](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/qrcode-ap.png)

3.  **Configure Wi-Fi:**
    * Open your browser and go to: **[http://reachy-mini.local:8000/settings](http://reachy-mini.local:8000/settings)**.
    * Enter your local Wi-Fi credentials (SSID & Password) and click **"Connect"**.
    * Wait a few moments for Reachy Mini to connect to your Wi-Fi network. The access point will disappear once connected. If the connection fails, Reachy Mini will restart the access point, and you can try again.

## 3. 🔄 Update System

Before going further, it is highly recommended to update your robot to the latest version.

1.  **Open Settings:** Go to **[http://reachy-mini.local:8000/settings](http://reachy-mini.local:8000/settings)**.
2.  **Check for Updates:** Click the **"Check for updates"** button.
3.  **Install:** If a new version is available, follow the on-screen instructions to install it.


## 4. 🕹️ Next Step: Using the Robot

Now that your robot is online and up to date, you can start controlling it!

👉 **[Go to the Usage Guide](usage.md)** to learn how to:
* Access the **Dashboard**.
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
