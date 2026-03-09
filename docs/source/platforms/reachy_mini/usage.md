# Using Reachy Mini

Now that your robot is connected, here is how to interact with it. You can control it visually using **Reachy Mini Control** or programmatically using **Python**.

## 1. Reachy Mini Control 🕹️

**Reachy Mini Control** is the desktop app for your robot. It allows you to check the robot's status, update the system, and manage applications.

**Download:** Get the latest version of **Reachy Mini Control** from the [official website](https://hf.co/reachy-mini/#/download).

**Open Reachy Mini Control** and connect to your robot. Once connected, you will see real-time information about your robot.

![Reachy Mini Control](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/control-app-dashboard.png)

### Features

* Control the **Head** and **Antennas** using the *Controller* tab.

![Controller Tab](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/control-app-controller.png)

* Play with **Expressions**: Make your robot happy, sad, angry, and more with the built-in expressions.

![Expressions Tab](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/control-app-expressions.png)

* In the *⚙️* tab, you can:
    * **System Updates:** Always keep your robot up to date.
    * **Network:** Configure Wi-Fi connections.

## 2. Applications 📱

Reachy Mini can run "Apps" — autonomous behaviors packaged for the robot (like a Conversation demo, a Game, or a Dance).

### How to use Apps
1.  **Browse:** Go to the *Applications* tab on Reachy Mini Control and click on "Discover Apps". This will open the Hugging Face Spaces ecosystem, where you can find compatible apps for your robot.
2.  **Install:** Click on the "Install" button on an app to add it to your robot.
3.  **Launch:** Click the "Start ▶️" button on an installed app. The robot will start the behavior immediately.
4.  **Stop:** Click the "Stop" ⏹️ button to stop the application.

> **Note:** When an App is running, it takes control of the robot. You cannot run Python scripts while an App is active.

## 3. Coding Quickstart 🐍

Ready to write your own logic? Reachy Mini is controlled via a simple Python SDK.

👉 **[Go to the SDK documentation](../../SDK/readme.md)** for a complete overview.

## ❓ Troubleshooting

Encountering an issue? 👉 **[Check the Troubleshooting & FAQ Guide](../../troubleshooting.md)**
