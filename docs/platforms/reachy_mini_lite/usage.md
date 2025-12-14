# Using Reachy Mini Lite

The Lite version relies on your computer to run its intelligence. The central hub for this is the **Reachy Mini Control** application.

## 1. Reachy Mini Control (Dashboard) 🖥️

When you open the application, you access the complete control panel for your robot.

* **Status & Visualizer (Left Panel):**
    * **3D View:** Shows the real-time position of the robot.
    * **Ready/Not Ready:** Indicates if the robot is correctly connected via USB.
    * **Sensors:** Monitor the microphone input and speaker volume.
    * **Logs:** See technical details and connection events at the bottom.

## 2. Applications & Demos 📱

You don't need to code to start having fun. The app comes with an integrated ecosystem.

### Quick Actions
Located at the bottom right, these are built-in demos ready to launch instantly:
* **Expressions:** Make Reachy express emotions (Happy, Sad, Angry, etc.).
* **Controller:** Teleoperate the robot using a game controller or sliders. 

### Installing New Apps
To extend Reachy's capabilities with community-created behaviors:
1.  **Discover:** Click the **"Discover apps"** button. This opens the Hugging Face Spaces store.
2.  **Install:** Select an app (like a Game or a Conversation demo) and click "Install".
3.  **Play:** Once installed, the app will appear in your "Applications" list. Simply click **"Play"** to start it.

> **Note:** When an App is running, it controls the robot. Stop the app before trying to run your own Python scripts.

## 3. Coding with Python 🐍

The Desktop App runs the **Daemon** (the background service) automatically. This means you can write Python scripts that connect directly to the app.

**1. Install the library:**
If you haven't already, install the SDK on your computer:
```bash
pip install reachy-mini
```

**2. Write your code:**
Create a file named hello_reachy.py and run it.

```python
from reachy_mini import ReachyMini
import time

with ReachyMini() as mini:
    # Check connection
    print(f"Connected! Robot state: {mini.state}")
    
    # Wiggle antennas
    print("Wiggling antennas...")
    mini.goto_target(antennas=[0.5, -0.5], duration=0.5)
    time.sleep(0.5)
    mini.goto_target(antennas=[-0.5, 0.5], duration=0.5)
```

👉 **[Go to the SDK Quickstart](../../sdk/quickstart.md)** for a complete tutorial.

## ❓ Troubleshooting

Encountering an issue? 👉 **[Check the Troubleshooting & FAQ Guide](/docs/troubleshooting.md)**