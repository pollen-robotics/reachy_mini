# Dashboard & Control (Lite)

Since the **Reachy Mini Lite** doesn't have an onboard computer, the "Dashboard" runs on your own computer.

## Reachy Mini Control App (Recommended)

The easiest way to use the robot is the **Reachy Mini Control** desktop application.

### 1. Installation
* [Download for Windows](#) (Link to be added)
* [Download for macOS](#) (Link to be added)
* [Download for Linux](#) (Link to be added)

### 2. Features
* **Connection Status:** Shows if the robot is correctly detected via USB.
* **App Launcher:** Run official demos and community apps locally.
* **Viewer:** Visualize the camera feed and motor positions.

## For Developers (Python SDK)

If you prefer using the command line or building your own scripts, you don't strictly need the desktop app. You can use the dashboard provided by the Python SDK.

1.  **Install the SDK:** `pip install reachy-mini`
2.  **Start the Daemon:** `reachy-mini-daemon`
3.  **Open Browser:** Go to `http://localhost:8000` to see the local web dashboard.