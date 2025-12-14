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

**1. Install the library on your computer:**
```bash
pip install reachy-mini
```

**2. Write your code:**
Since your robot is on the network, you can control it remotely!

```python
from reachy_mini import ReachyMini

# Replace with your robot's IP if 'reachy-mini.local' doesn't resolve
with ReachyMini(host="reachy-mini.local") as mini:
    mini.media.speak("Hello, world!")
    mini.turn_on("fan")
```

👉 **[Go to the SDK Quickstart](../../sdk/quickstart.md)** for a complete tutorial.

---

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

# Connects to localhost by default
with ReachyMini() as mini:
    print("I am connected via USB!")
    mini.media.speak("Ready to code.")
```

👉 **[Go to the SDK Quickstart](../../sdk/quickstart.md)** for a complete tutorial.

---

# Using the Simulation

The simulation provides the exact same interface as the real robot. This means you can use the Dashboard and Python SDK exactly as if you had a Reachy Mini Lite connected.

## 1. The Dashboard 🕹️

When you run the simulation daemon:
```bash
reachy-mini-daemon --sim
```

You can access the Dashboard at **[http://localhost:8000](http://localhost:8000)**.
* **Visualize:** See the camera feed (simulated view).
* **Apps:** You can install and run Apps! They will execute inside the simulation (e.g., the robot will dance in the 3D viewer).

## 2. Coding Quickstart 🐍

Your Python code works seamlessly with the simulation.

**1. Install:**
```bash
pip install reachy-mini[mujoco]
```

**2. Run:**
```python
from reachy_mini import ReachyMini

# Connects to the simulated robot
with ReachyMini() as mini:
    # This movement happens in the MuJoCo window!
    mini.media.speak("I am alive... virtually.")
```

👉 **[Go to the SDK Quickstart](../../sdk/quickstart.md)** for a complete tutorial.