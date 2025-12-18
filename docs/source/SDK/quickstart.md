# Quickstart Guide

Follow this guide to get your Reachy Mini up and running, either on real hardware or in simulation.

## 1. Prerequisites

You have correctly installed Reachy Mini on your computer, see [this guide](/docs/SDK/installation.md).

## 2. Ensure the Robot Server is running (Daemon)

The **Daemon** is a background service that handles the low-level communication with motors and sensors. It must be running for your code to work.

* **On Reachy Mini (Wireless)**: The daemon is running when the robot is powered on. Ensure your computer and Reachy Mini are on the same network.
* **On Reachy Mini Lite (USB)**: You have two options
  - Start the [desktop application](../reachy_mini_lite/get_started_lite.md)
  - Open a terminal and run `reachy-mini-daemon`
* **For Simulation (No robot needed):** You have two options
  - Start the [desktop application](../reachy_mini_lite/get_started_lite.md)
  - Open a terminal and run `reachy-mini-daemon --sim`


✅ **Verification:** Open [http://localhost:8000](http://localhost:8000) in your browser. If you see the Reachy Dashboard, you are ready!

## 3. Your First Script

Keep the daemon terminal open. In a **new terminal**, create a file named `hello.py`:

> Tip: If you are using a Reachy Mini Wireless and running the script on your computer, you need to do `ReachyMini(localhost_only=False)`
```python
from reachy_mini import ReachyMini

# Connect to the running daemon
with ReachyMini() as mini:
    print("Connected to Reachy Mini! ")
    
    # Wiggle antennas
    print("Wiggling antennas...")
    mini.goto_target(antennas=[0.5, -0.5], duration=0.5)
    mini.goto_target(antennas=[-0.5, 0.5], duration=0.5)
    mini.goto_target(antennas=[0, 0], duration=0.5)

    print("Done!")
```

Run it:
```bash
python hello.py
```

## ❓ Troubleshooting
Encountering an issue? 👉 **[Check the Troubleshooting & FAQ Guide](../troubleshooting.md)**


## Next Steps
* **[Python SDK](python-sdk.md)**: Learn to move, see, speak, and hear.
* **[Browse the Examples Folder](https://github.com/pollen-robotics/reachy_mini/tree/develop/examples)**
* **[AI Integrations](integration.md)**: Connect LLMs, build Apps, and publish to Hugging Face.
* **[Core Concepts](core-concept.md)**: Architecture, coordinate systems, and safety limits.
