# Using Reachy Mini Lite

Now that your robot is connected, here is how to interact with it. You can control it visually using **Reachy Mini Control** or programmatically using **Python**.

Check [this guide](./get_started.md) if you have not installed the app yet.

## 1. Reachy Mini Control 🖥️

**Reachy Mini Control** is the desktop app for your robot. It allows you to check the robot's status, update the system, and manage applications.

Open **Reachy Mini Control** and connect your robot via USB. Once connected, you will see real-time information about your robot.

* **Status & Visualizer (Left Panel):**
    * **3D View:** Shows the real-time position of the robot.
    * **Ready/Not Ready:** Indicates if the robot is correctly connected via USB.
    * **Sensors:** Monitor the microphone input and speaker volume.
    * **Logs:** See technical details and connection events at the bottom.

### Features

* Control the **Head** and **Antennas** using the *Controller* tab.

![Controller Tab](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/control-app-controller.png)

* Play with **Expressions**: Make your robot happy, sad, angry, and more with the built-in expressions.

![Expressions Tab](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/control-app-expressions.png)

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
