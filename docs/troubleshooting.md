# Troubleshooting & FAQ

Welcome to the Reachy Mini support page. Click on the questions below to reveal the answers.

##  Known Issues - Batch December 2025

<details>
<summary><strong>The head may touch the body during some official motions</strong></summary>

This behavior is expected and not a hardware or software bug.  
However, since it can be confusing, we will update those motions to avoid this contact.

</details>

<details>

<summary><strong>Low audio volume</strong></summary>

- Update your robot to version 1.2.3 or later

For more details, see the documentation:  
[Getting Started](/docs/platforms/reachy_mini/get_started.md)

</details>

<details>
<summary><strong>Permission errors</strong></summary>

- Update your robot to version 1.2.3 or later  
- Reboot the robot

</details>

<details>
<summary><strong>An antenna appears rotated by 90° or 180°</strong></summary>

This is a manufacturing issue.

It is easy to fix by following this guide:  
[Antenna repositioning guide](https://cdn.discordapp.com/attachments/1450364269886705757/1450801804374970529/reachy-mini-antenna-mismounted-v3.pdf?ex=69452d78&is=6943dbf8&hm=1c870bd64b3e476501b665c83c7de9412a4495f8e059bb3a9f3d7fd1f5e188f1&)

</details>

<details>
<summary><strong>[Wireless] The wifi access point doesn't appear and the fan keeps spinning</strong></summary>

When starting the robot, you should hear a fan spinning. It should stop spinning after a few seconds. If it doesn't stop spinning, it means the OS is not booting. This probably means the switch on the head board is in the `DOWNLOAD SW1` position.

![Head board switch](/docs/assets/head_board_switch.png)

It should be in the `DEBUG` position.

</details>


##  Most commons questions

<details>
<summary><strong>I have 2 cables and a few screws left after finishing the assembly. Is this normal?</strong></summary>

Yes, this is completely normal.  
We intentionally include spare cables and screws in the kit in case some parts are damaged or lost during assembly.

You do not need to install them.

</details>

<details>
<summary><strong>Image is dark on the Lite version</strong></summary>

- set auto-exposure-priority=1 using uvc-util on macOS

</details>

<details>
<summary><strong>Motor blinking red or Overload Error</strong></summary>

If you get "Motor hardware errors: ['Overload Error']" a few second after starting the robot for the first time.

It is VERY likely there are motor not in the good slot, e.g motor 4 on slot 3


- [**Reachy Mini Wireless - Step-by-Step Guide**](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini_Assembly_Guide)
- [**Reachy Mini LITE - Step-by-Step Guide**](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini_LITE_Assembly_Guide)


</details>



## 📋 Table of Contents

- [Troubleshooting \& FAQ](#troubleshooting--faq)
  - [Known Issues - Batch December 2025](#known-issues---batch-december-2025)
  - [Most commons questions](#most-commons-questions)
  - [📋 Table of Contents](#-table-of-contents)
  - [🚀 Getting Started \& Assembly](#-getting-started--assembly)
  - [🔌 Connection \& Dashboard](#-connection--dashboard)
  - [🤖 Hardware, Motors \& Limits](#-hardware-motors--limits)
  - [🐍 SDK, Apps \& Programming](#-sdk-apps--programming)
  - [🕹️ Moving the Robot](#️-moving-the-robot)
  - [👁️ Vision \& Audio](#️-vision--audio)
  - [🔧 Specific Error Messages \& Fixes](#-specific-error-messages--fixes)
  - [📦 Shipping \& Warranty](#-shipping--warranty)
  - [💬 Still stuck?](#-still-stuck)

<br>

## 🚀 Getting Started & Assembly

<details>
<summary><strong>How long does assembly usually take?</strong></summary>

Most testers report between **1.5 and 2 hours**. It can take up to 4 hours depending on your experience level.

</details>

<details>
<summary><strong>Are there any difficult steps during assembly?</strong></summary>

Testers generally describe it as fun and simple. The trickiest parts are usually:
* **Cable routing:** ensuring cables aren't pinched.
* **Torque:** Tightening parts correctly (not too loose, not too hard).

</details>

<details>
<summary><strong>My Reachy Mini doesn’t move on first startup. What should I check?</strong></summary>

* **Power Supply:** Ensure the 7V-5A power supply is plugged in. The USB connection is not enough to power the motors.
* **Cables:** Check that all cables are fully inserted. Loose power cables are a common cause of "motor not responding" errors.

</details>

<details>
<summary><strong>Do I need to start the daemon manually?</strong></summary>
**NO** 

- With Reachy Mini (Wireless), the daemon is already running on the embedded Raspberry Pi.
- With Reachy Mini Lite, you can use [the desktop app](/docs/platforms/reachy_mini_lite/get_started.md).

</details>

<br>

## 🔌 Connection & Dashboard

<details>
<summary><strong>How do I connect the robot to Wi-Fi?</strong></summary>

See : [Reachy Mini Wireless guide](/docs/platforms/reachy_mini/get_started.md) .



</details>

<details>
<summary><strong>The dashboard at http://localhost:8000 doesn't work.</strong></summary>

Perform these checks:
1.  **Virtual Environment:** Ensure you are running inside your virtual environment (`.venv`, `reachy_mini_env`,...).
2.  **SDK Update:** Ensure you have the latest version.
- With `pip`, run :
```bash
pip install -U reachy-mini
```
- With `uv`, run :
```bash
uv add reachy-mini
```
1.  **Daemon:** Make sure the daemon `reachy-mini-daemon` is running in a terminal.

</details>

<details>
<summary><strong>Does the robot have a Web API?</strong></summary>

Yes. The daemon provides a REST API (FastAPI) and WebSocket support.
* **Docs:** `http://localhost:8000/docs` (available when daemon is running).
* **Features:** Get state, Move joints, Control daemon.

</details>

<details>
<summary><strong>All apps installations fail on Windows !</strong></summary>

It might be related to unsufficient rights to create symlinks in Windows. You can set the environment variable `HF_HUB_DISABLE_SYMLINKS_WARNING` to 1 to remove the warnings that cause the failure.

In a terminal, run :
```powershell
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

</details>

<br>

## 🤖 Hardware, Motors & Limits

<details>
<summary><strong>What are the safety limits (Head & Body)?</strong></summary>

If you command a pose outside these limits, the robot will automatically clamp to the nearest safe pose.

* **Body Yaw:** [-180°, 180°].
* **Head Pitch/Roll:** [-40°, 40°].
* **Head Yaw:** [-180°, 180°].
* **Combined Limit:** The difference between `body_yaw` and `head_yaw` must be within **[-65°, 65°]**.

</details>

<details>
<summary><strong>Why are the motors "limp" or "stiff"? (Compliancy)</strong></summary>

* **`enable_motors()`**: Motors **ON** (Stiff). Robot holds position.
* **`disable_motors()`**: Motors **OFF** (Limp). You can move it by hand.
* **`make_motors_compliant()`**: Motors **ON but Soft**. Useful for teaching-by-demonstration.

</details>

<details>
<summary><strong>Motors stop responding after a while.</strong></summary>

* Check the power supply connection.
* Motors might have entered thermal protection mode (overheating). Turn off and on again.
* Updating the SDK (`pip install -U reachy-mini`) has solved this for some users.

</details>

<details>
<summary><strong>Can I modify the appearance (Skins/CAD)?</strong></summary>

* **CAD:** Not currently public.
* **Skins:** Yes, the community has created custom builds (e.g., Star Wars droids).

</details>

<br>

## 🐍 SDK, Apps & Programming

<details>
<summary><strong>How do I connect from Python?</strong></summary>

Use the `ReachyMini` class.

```python
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    # Your code here
    print(mini.state)
```

</details>

<details>
<summary><strong>How do I create a new App?</strong></summary>

1.  Use the generator: `reachy-mini-make-app my_app_name`.
2.  Edit `main.py` in the generated folder.
3.  Run it: `python my_app_name/main.py`.

Check the [Hugging Face Tutorial](https://huggingface.co/blog/pollen-robotics/make-and-publish-your-reachy-mini-apps) for details.

</details>

<details>
<summary><strong>Installing apps from Hugging Face fails.</strong></summary>

Update your SDK. Early versions had a bug with Space installation.

```bash
pip install -U reachy-mini
```

</details>

<details>
<summary><strong>Is there a Simulation mode?</strong></summary>

Yes, via MuJoCo. It is still a work in progress, but you can run code with the `--sim` flag or `ReachyMini(media_backend="no_media")` if just testing logic without physics.

</details>

<br>

## 🕹️ Moving the Robot

<details>
<summary><strong>How do I move the head?</strong></summary>

Use `goto_target` with `create_head_pose`:

```python
from reachy_mini.utils import create_head_pose

# ... inside with ReachyMini() as mini:
mini.goto_target(head=create_head_pose(yaw=-10, pitch=20))
```

</details>

<details>
<summary><strong>What is the difference between `goto_target` and `set_target`?</strong></summary>

* **`goto_target`**: **Smooth**. Interpolates motion over time (default 0.5s). Best for gestures.
* **`set_target`**: **Instant**. Sets the target immediately. Best for high-frequency control (teleoperation, mathematical trajectories).

</details>

<details>
<summary><strong>How do I record and replay moves?</strong></summary>

**Recording:**
Call `start_recording()` and `stop_recording()` around your control loop.

```python
mini.start_recording()
# ... move robot ...
move = mini.stop_recording()
```

**Replaying:**
Use the `RecordedMoves` class to load moves from the [Hugging Face library](https://huggingface.co/pollen-robotics/reachy-mini-dances-library).

```python
mini.play_move(recorded_moves.get("dance_1"))
```

</details>

<br>

## 👁️ Vision & Audio

<details>
<summary><strong>Volume is too low (Linux)</strong></summary>

1.  Run `alsamixer`.
2.  Set **PCM1** to 100%.
3.  Use **PCM,0** to adjust the global volume.

To make it permanent:
```bash
CARD=$(aplay -l | grep -i "reSpeaker" | head -n1 | sed -n 's/^card \([0-9]*\):.*/\1/p')
amixer -c "$CARD" set PCM,1 100%
sudo alsactl store "$CARD"
```

</details>

<details>
<summary><strong>How do I get camera frames?</strong></summary>

Use the `media` object.

```python
with ReachyMini() as mini:
    frame = mini.media.get_frame()
    # Returns an OpenCV-compatible numpy array
```

</details>

<details>
<summary><strong>How do I use the Microphone / Speaker?</strong></summary>

```python
# Get audio 
sample = mini.media.get_audio_sample()

# Play audio
mini.media.push_audio_sample(numpy_chunk)
```

</details>

<details>
<summary><strong>How do I make Reachy look at something?</strong></summary>

* **2D (Image):** `mini.look_at_image(x, y)` - (0,0 is top-left).
* **3D (World):** `mini.look_at_world(x, y, z)` - Coordinates in robot frame.

</details>

<details>
<summary><strong>Face tracking feels slow.</strong></summary>

Performance relies heavily on lighting conditions. Ensure the face is well-lit. Using the GStreamer backend can also improve latency compared to the default OpenCV backend.

</details>

<br>

## 🔧 Specific Error Messages & Fixes

<details>
<summary><strong>Motor '<name>' hardware errors: ['Input Voltage Error']</strong></summary>
We are using a higher voltage on Reachy Mini, it's on purpose :)

</details>



<details>
<summary><strong>Error: "OSError: PortAudio library not found"</strong></summary>

You are missing a system dependency. Run:

```bash
sudo apt-get install libportaudio2
```
Then restart the daemon.

</details>

<details>
<summary><strong>Warning: "Circular buffer overrun" (Simulation/Mujoco)</strong></summary>

This appears if you connect to the robot but don't consume the video frames, causing the buffer to fill up.
* **Fix:** If you don't need video, initialize with `ReachyMini(media_backend="no_media")`.

</details>

<details>
<summary><strong>No Microphone Input / Direction of Arrival (Beta Units)</strong></summary>

* **No Input:** Requires firmware 2.1.3. Run the [update script](../src/reachy_mini/assets/firmware/update.sh).
* **No Direction:** Requires firmware 2.1.0+.
* Check flex cables (Slides 45-47 of assembly guide).

</details>

<br>

## 📦 Shipping & Warranty

<details>
<summary><strong>My package is damaged or missing.</strong></summary>

Contact **Pollen Robotics** team immediately. You can send us an email to sales@pollen-robotics.com with photos of the package, receipt number or invoice number and your full name. We will then check with the transport company and keep you updated.

</details>

<details>
<summary><strong>Refund Policy</strong></summary>

* **Before shipping:** Contact `sales@pollen-robotics.com` for a 100% refund.
* **After shipping:** You have 30 days to return your package. Contact sales (sales@pollen-robotics.com) with proof of delivery and invoice or receipt number. If you have comments / feedback, please let us know, our focus is building a robot the open-source community enjoys building. 

</details>

<details>
<summary><strong>Warranty</strong></summary>

If a part is broken/malfunctioning, Pollen's after-sales team will determine if it is a hardware defect covered by warranty. Then, our manufacturer will provide repair or replacement parts. You can send us an email to sales@pollen-robotics.com with photos of the issue, receipt number or invoice number and your full name.

</details>


## 💬 Still stuck?

If you couldn't find the answer to your issue in this guide, please reach out to us directly!
The Pollen Robotics team and the community are active on Discord to help you troubleshoot specific problems.

👉 **[Join the Pollen Robotics Discord](https://discord.gg/Y7FgMqHsub)**
