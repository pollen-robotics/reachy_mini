# Build with Reachy Mini

Create amazing experiences with Python or JavaScript. Control movements, access sensors, build apps, and share them with the community.

> **⚠️ Before you start**
>
> Make sure you have assembled your Reachy Mini and connected it. If not, follow the **Getting Started guide** first.

---

## Step 1: Install the SDK

The Reachy Mini SDK is a Python package that lets you control the robot programmatically. It includes both the daemon (background service) and the control API.

### Prerequisites
* **Python 3.10 - 3.13** (We recommend using a virtual environment)
* **Git LFS** (Required for large model files)
* **Reachy Mini connected** (USB for Lite or Wi-Fi for Wireless version)

### Install Git LFS
* **Linux:** `sudo apt install git-lfs`
* **macOS:** `brew install git-lfs`
* **Windows:** Download from [git-lfs.com](https://git-lfs.com)

### Install Reachy Mini

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install the SDK
pip install reachy-mini
```

> **Note for `uv` users:** You can run `uv run reachy-mini-daemon` directly without manual setup.

<details>
<summary><strong>🐧 Linux: Set up udev rules (required for USB connection)</strong></summary>

```bash
# Create udev rules
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", GROUP="dialout"
SUBSYSTEM=="tty", ATTRS{idVendor}=="38fb", ATTRS{idProduct}=="1001", MODE="0666", GROUP="dialout"' \
| sudo tee /etc/udev/rules.d/99-reachy-mini.rules

# Reload rules and add user to dialout group
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG dialout $USER

# Log out and back in for changes to take effect
```
</details>

---

## Step 2: Your First Script

Before running any script, you need to start the daemon. The daemon handles communication with the robot's motors and sensors.

### 1. Start the Daemon
Open a terminal and run:

```bash
reachy-mini-daemon
```

*For simulation (no robot needed):*
```bash
pip install reachy-mini[mujoco]
reachy-mini-daemon --sim
```

### 2. Run Your First Script
In another terminal, create `hello.py`:

```python
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    print("Connected!")
    print(f"Robot state: {mini.state}")
```

Run it:
```bash
python hello.py
```

---

## Step 3: Control the Robot

Use `goto_target` for smooth movements and `set_target` for immediate control.

### Move the Head

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as mini:
    # Move head up and tilt
    pose = create_head_pose(
        z=10,      # 10mm up
        roll=15,   # 15° tilt
        degrees=True, 
        mm=True
    )
    mini.goto_target(head=pose, duration=2.0)
    
    # Reset to default
    mini.goto_target(
        head=create_head_pose(), 
        duration=1.0
    )
```

### Control Antennas & Body

```python
import numpy as np
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as mini:
    # Move everything at once
    mini.goto_target(
        head=create_head_pose(y=-10, mm=True),
        antennas=np.deg2rad([45, 45]),
        body_yaw=np.deg2rad(30),
        duration=2.0,
    )
    
    # Make antennas wave
    for _ in range(3):
        mini.goto_target(
            antennas=np.deg2rad([60, 20]),
            duration=0.3
        )
        mini.goto_target(
            antennas=np.deg2rad([20, 60]),
            duration=0.3
        )
```

### Motor Control Modes

* **`enable_motors()`**: Powers motors ON. Robot holds position firmly.
* **`disable_motors()`**: Powers motors OFF. Robot is completely limp.
* **`make_motors_compliant()`**: Motors ON but soft. Great for teaching-by-demonstration.

---

## Step 4: Access Sensors

Reachy Mini has a camera, microphones, speaker, and accelerometer (wireless version).

### 📷 Camera

```python
import cv2
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    # Get a frame (numpy array)
    frame = mini.media.get_frame()
    
    # Display with OpenCV
    cv2.imshow("Reachy View", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Or save to file
    cv2.imwrite("snapshot.jpg", frame)
```

### 🎙️ Microphone

```python
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    # Get audio sample
    sample = mini.media.get_audio_sample()
    
    # sample is a numpy array
    # Process with your favorite 
    # audio/speech library
```

### 🔊 Speaker

```python
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    # Play audio file
    mini.media.play_audio("hello.wav")
    
    # Or use TTS (with external lib)
    # and play the generated audio
```

### 📏 Accelerometer (Wireless only)

```python
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    # Get accelerometer data
    accel = mini.sensors.accelerometer
    
    print(f"X: {accel.x}")
    print(f"Y: {accel.y}")
    print(f"Z: {accel.z}")
```

---

## Step 5: Create an App

Package your code as a Reachy Mini app that can be installed from the dashboard.

### Generate App Template
Use the built-in generator to create a complete project structure:

```bash
reachy-mini-make-app my_awesome_app
```

This creates: `pyproject.toml`, `README.md`, and the main app file.

### App Structure (`my_awesome_app/app.py`)

```python
from reachy_mini.apps.app import ReachyMiniApp
from reachy_mini import ReachyMini

class MyAwesomeApp(ReachyMiniApp):
    """My first Reachy Mini app!"""
    
    def run(self, reachy_mini: ReachyMini, stop_event):
        """Main app logic. Called when the app starts."""
        
        while not stop_event.is_set():
            # Your code here
            # Check stop_event periodically to allow clean shutdown
            
            # Example: wave antennas
            reachy_mini.goto_target(
                antennas=[0.5, -0.5],
                duration=0.5
            )
            reachy_mini.goto_target(
                antennas=[-0.5, 0.5],
                duration=0.5
            )
```

---

## Step 6: Publish to Hugging Face

Share your app with the community on Hugging Face Spaces. Apps published there can be installed directly from the Reachy Mini dashboard.

1.  **Create a Space**
    * Go to [Hugging Face New Space](https://huggingface.co/new-space).
    * Choose "Gradio" or "Static" as the SDK.

2.  **Add the `reachy_mini` tag**
    * In your Space's `README.md`, add this tag to the metadata:

```yaml
---
title: My Awesome App
emoji: 🤖
tags:
  - reachy_mini  # This makes it discoverable!
---
```

**That's it!** Your app will now appear in the Reachy Mini dashboard's app store.

---

## REST API

Prefer HTTP? The daemon exposes a REST API for language-agnostic control.

**Get Robot State:**
```bash
curl http://localhost:8000/api/state/full
```

**API Documentation:**
Full OpenAPI docs are available when the daemon is running at:
`http://localhost:8000/docs`

---

## Ready to explore more?

Check out the full documentation and community apps for inspiration.

* [Full SDK Documentation](https://github.com/pollen-robotics/reachy_mini/blob/main/docs/python-sdk.md)
* Browse Community Apps
* FAQ