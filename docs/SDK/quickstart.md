# Quickstart Guide

Follow this guide to get your Reachy Mini up and running, either on real hardware or in simulation.

## 1. Prerequisites

Before installing the python package, ensure you have the system dependencies:

* **Python 3.10 - 3.13** (Virtual environment recommended)
* **Git LFS**: Required for downloading large model files.
    * *Linux:* `sudo apt install git-lfs`
    * *macOS:* `brew install git-lfs`
    * *Windows:* [Download Installer](https://git-lfs.com)

## 2. Install the Software

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install the SDK
pip install reachy-mini
```

> **Note for `uv` users:** You can skip the venv setup and run commands directly with `uv run ...`.

### 🐧 Linux Users: USB Permission Setup
If you are on Linux and using the robot via USB, you must set up udev rules:

<details>
<summary>Click to see udev instructions</summary>

```bash
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", GROUP="dialout"
SUBSYSTEM=="tty", ATTRS{idVendor}=="38fb", ATTRS{idProduct}=="1001", MODE="0666", GROUP="dialout"' \
| sudo tee /etc/udev/rules.d/99-reachy-mini.rules

sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG dialout $USER
# Log out and log back in!
```
</details>

## 3. Start the Robot Server (Daemon)

The **Daemon** is a background service that handles the low-level communication with motors and sensors. It must be running for your code to work.

Open a terminal and run:

**For Real Robot:**
```bash
reachy-mini-daemon
```

**For Simulation (No robot needed):**
```bash
reachy-mini-daemon --sim
```

✅ **Verification:** Open [http://localhost:8000](http://localhost:8000) in your browser. If you see the Reachy Dashboard, you are ready!

## 4. Your First Script

Keep the daemon terminal open. In a **new terminal**, create a file named `hello.py`:

```python
from reachy_mini import ReachyMini
import time

# Connect to the running daemon
with ReachyMini() as mini:
    print(f"Connected! Robot State: {mini.state}")
    
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