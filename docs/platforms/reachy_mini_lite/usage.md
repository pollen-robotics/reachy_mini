# Using Reachy Mini Lite

The Lite version relies on your computer to run its intelligence. You can use it through a user-friendly **Desktop App** or in **Developer Mode**.

## 1. Reachy Mini Control (Desktop App) 🖥️

For beginners and demos, we recommend the **Reachy Mini Control** application.

* **Download:** [Windows](#) | [macOS](#) | [Linux](#)
* **Features:**
    * **Visualizer:** See the camera feed and robot posture.
    * **App Launcher:** Run official demos (Teleoperation, Dances) with one click.
    * **Status:** See connection health.

## 2. Developer Mode (Web Dashboard) 🛠️

If you want to run Python scripts or access advanced features, you will use the **Daemon**.

**1. Start the Daemon:**
Open a terminal and run:
```bash
reachy-mini-daemon
```

**2. Open the Dashboard:**
Go to [http://localhost:8000](http://localhost:8000).
* **Monitor:** Check motor states.
* **Apps:** You can also install/run apps from Hugging Face here, just like on the Wireless version!

## 3. Coding Quickstart 🐍

To control the robot with your own code:

**1. Install the library:**
```bash
pip install reachy-mini
```

**2. Write your code:**

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