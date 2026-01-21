# Using Reachy Mini

Now that your robot is connected, here is how to interact with it. You can control it visually using the **Dashboard** or programmatically using **Python**.

## 1. The Dashboard 🕹️

The Dashboard is the web interface running inside your robot. It allows you to check the robot's status, update the system, and manage applications.

**Access:** Open [http://reachy-mini.local:8000/](http://reachy-mini.local:8000/) in your browser.

### Features
* **System Updates:** Always keep your robot up to date. Go to the *Settings* tab and click "Check for updates".
* **Hardware Monitor:** Check battery level, motor temperatures, and disk usage.
* **Network:** Configure Wi-Fi connections.

## 2. Applications 📱

Reachy Mini can run "Apps" — autonomous behaviors packaged for the robot (like a Conversation demo, a Game, or a Dance).

### How to use Apps
1.  **Browse:** Go to the *Apps* tab on the Dashboard.
2.  **Install:** Click on the "Store" button to browse the [Hugging Face Spaces](https://huggingface.co/spaces?q=reachy_mini) ecosystem. You can install any compatible app with one click.
3.  **Launch:** Click the "Play" ▶️ button on an installed app. The robot will start the behavior immediately.
4.  **Stop:** Click the "Stop" ⏹️ button to kill the application.

> **Note:** When an App is running, it takes control of the robot. You cannot run Python scripts while an App is active.

## 3. Coding Quickstart 🐍

Ready to write your own logic? Reachy Mini is controlled via a simple Python SDK.

👉 **[Go to the SDK documentation](../../SDK/readme.md)** for a complete overview.

## ❓ Troubleshooting

Encountering an issue? 👉 **[Check the Troubleshooting & FAQ Guide](../../troubleshooting.md)**
