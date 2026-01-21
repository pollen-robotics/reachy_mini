# Quickstart Guide

Follow this guide to get your Reachy Mini up and running, either on real hardware or in simulation.

## 1. Prerequisites

Make sure you have installed Reachy Mini on your computer following our [installation guide](installation.md). 

> **💡 Important:** Ensure you have created and activated your Python virtual environment. **Remember to activate it every time you open a new terminal!**

<details>
<summary><strong>🔧 Run SDK directly on Wireless Reachy Mini (Optional)</strong></summary>

If you want to run the SDK directly on your wireless Reachy Mini instead of remotely on your computer, you can connect via SSH into it.

### Step 1: SSH Connection

Open a terminal and run:

```bash
ssh pollen@reachy-mini
```

When prompted, use these **default credentials**:
- Username: `pollen`
- Password: `root`

### Step 2: Activate the Python virtual environment

After connecting, activate the virtual environment:

```bash
source /venvs/apps_venv/bin/activate
```

### Step 3: Run scripts locally

When running scripts on Reachy Mini itself, use the standard `ReachyMini()` constructor. Auto mode will keep the connection on localhost unless you explicitly override it:

```python
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    # Your code here
```

> **💡 Benefits of running locally:** Lower latency, no network dependency, and direct access to all robot resources.
> 
> **⚠️ Drawbacks:** Lower CPU power, and no GUI available.


</details>

## 2. Ensure the Robot Server is running (Daemon)

The **Daemon** is a background service that handles the low-level communication with motors and sensors. It must be running for your code to work.

* **On Reachy Mini (Wireless)**: The daemon is running when the robot is powered on. Ensure your computer and Reachy Mini are on the same network.
* **On Reachy Mini Lite (USB)** - You have two options :
  - Start the [desktop application](../platforms/reachy_mini_lite/get_started.md#3--download-reachy-mini-control)
  - Open a terminal and run :
  ```bash
  uv run reachy-mini-daemon
  ```
* **For Simulation (No robot needed)** - You have two options :
  - Start the [desktop application](../platforms/reachy_mini_lite/get_started.md#3--download-reachy-mini-control)
  - Open a terminal and run:
    - **Linux/Windows:**
      ```bash
      uv run reachy-mini-daemon --sim
      ```
    - **macOS:**
      ```bash
      mjpython -m reachy_mini.daemon.app.main --sim
      ```
      > **⚠️ macOS Users:** `uv` may have compatibility issues with MuJoCo on macOS. If you encounter installation or runtime problems, it's recommended to use `pip` directly instead of `uv` for MuJoCo-related packages.

✅ **Verification:** Open [http://localhost:8000](http://localhost:8000) in your browser. If you see the Reachy Dashboard, you are ready!

## 3. Your First Script

> **⚠️ Important:** Keep the daemon terminal open and running! The daemon must stay active for your robot to work.

### Create your Python script

**Step 1:** Open a new terminal window

**Step 2:** Create a new file called `hello.py` and copy-paste the following code into it:

> Tip: The constructor now auto-detects Lite vs Wireless and switches between localhost and network automatically. Only override it for advanced cases, e.g. `ReachyMini(connection_mode="network")`.
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

**Step 3:** Save the file and run your script:

In your new terminal, run:
```bash
python hello.py
```

🎉 If everything went well, your robot should now wiggle its antennas!

## ❓ Troubleshooting
Encountering an issue? 👉 **[Check the Troubleshooting & FAQ Guide](../troubleshooting.md)**


## Next Steps
* **[Python SDK](python-sdk.md)**: Learn to move, see, speak, and hear.
* **[Browse the Examples Folder](https://github.com/pollen-robotics/reachy_mini/tree/main/examples)**
* **[AI Integrations](integration.md)**: Connect LLMs, build Apps, and publish to Hugging Face.
* **[Core Concepts](core-concept.md)**: Architecture, coordinate systems, and safety limits.
