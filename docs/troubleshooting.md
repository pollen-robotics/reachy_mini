# Troubleshooting & FAQ

Welcome to the Reachy Mini support page. Click on the questions below to reveal the answers.

##  Troubleshooting - Batch December 2025
### Essential troubleshooting - Please read this first, it solves all well-known issues!


<details><summary><strong>Before anything else and for any issue: update & restart</strong></summary>

**Make sure you are using up-to-date software and that you have restarted both your robot and your computer.**  
To restart your robot, press OFF, wait 5 seconds, then press ON. This simple procedure fixes several common and well-known issues.

**How to update the software:**

- **If you are using the dashboard in a web browser**  
  Open `Settings`, then click **Check for updates**.  
  ![Update](/docs/assets/update.png)
- **If you are using the new dashboard/app**  
  Since 0.8.5 of the app, 
- **If you are using a cloned repository**  
  Make sure you are either:
  - On the latest tagged release, or
  - Up to date with the `develop` branch (`git pull`).
</details>


<details><summary><strong>Motor blinking red or Overload Error</strong></summary>

**1. Motors inversion**: If you get "Motor hardware errors: ['Overload Error']" a few second after starting the robot **for the first time.** and have two motors arm pointing upward.  
It is VERY likely there are motors not placed in the good slot, e.g motor 1 on slot 2.

<details><summary>See illustration</summary>

![Motors inversion symptom](/docs/assets/motors_upward.png)

</details>

Check assembly guide:

- [**Reachy Mini Wireless - Step-by-Step Guide**](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini_Assembly_Guide)
- [**Reachy Mini LITE - Step-by-Step Guide**](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini_LITE_Assembly_Guide)

**2. Check the arm orientation on the motor's horn**:
Remove the faulty motor, then place the arm upward like in the attached picture. Then check if you can see the the two line marks aligned as represented:

<details><summary>See picture:</summary>

![Marks_aligned](/docs/assets/marks_aligned.png)
</details> 

If they are not, please remove the two screws securing the arm and put it back with the two lines matching.  


**3. check the extra length of the usb cable inside the head:**  
If it's too long inside the head, there must miss some slack underneath and the head cannot move freely.  
So the motors force too much and can be damaged.  
<details><summary>See picture:</summary>

![usb_cable_length](/docs/assets/usb_cable_length.jpg)
</details>  

Please let some slack to the usb cable to allow the head to move freely, even to its maximal height position.  



**4. A motor feels broken:**
We identified an issue affecting a limited production batch of Reachy Mini robots, related to a faulty batch of Dynamixel motor. 

In most reported cases, the issue affects motor number 4 or one with QC label n°2544.

If one of your motors, feels blocked or unusually hard to move, when turned off [(example video here)](https://drive.google.com/file/d/1UHTqUcb21aFThqlr2Qcx23VWqvj_y-ly/view?usp=sharing), and you are 100% sure the motor was in the correct slot.

It's probably a broken motor.

First, try to update your robot to the latest software version, then reboot it. This will reflash your motors.
If the issue persists, please fill out this short form so we can track and ship you a new motor:  https://forms.gle/JdhMzadeCnbynw7Q6
</details>


<details>
<summary><strong>A motor is not moving at all, but get stiff when powered on, and doesn't blink red </strong></summary>

This behavior happen when a motor (often n°1) has not been flashed properly during the manufacturing process.  
=> Please power your robot but don't turn it on with the dashboard/daemon, then update reachy mini's software, then reboot the robot. This will reflash your motors.

</details>

<details>
<summary><strong>Electrical Shock Error </strong></summary>

An electrical shock error on Dynamixel motors means there is either an issue with the power supply, or a short circuit somewhere.
Please check if any cable is damaged, from the foot PCB to the head. Especially the followings cables:  
- Power Cable (black & red) 
- 3-wires cables for motors (300mm, 200mm, 100mm and 40mm)

It can also be the same issue as "Motor blinking red or Overload Error" described above.

</details>

<details>

<summary><strong>Missing Motor Error / No motor found on port</strong></summary>

- Make sure you have plugged all the motor cables correctly.
- Make sure you have every motor and not two same motor in the kit. Refer to the label on each motor. e.g motor 1, motor 2, motor 3, motor 4, L motor, R motor...  

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
[Antenna repositioning guide](https://drive.google.com/file/d/1FsmNpwELuXUbdhGHDMjG_CNpYXOMtR7A/view?usp=drive_link)

</details>

<details>
<summary><strong>Image is dark on the Lite version</strong></summary>

- set auto-exposure-priority=1 using uvc-util on macOS

</details>

<details>
<summary><strong>A part is missing in my package</strong></summary>

Be sure to unpack everything first. Some parts are pre-assembled (e.g the bottom head part is already placed in the back head part).

![head_parts](/docs/assets/head_parts.jpg)

Then, check the assembly guide's parts list to see if you really miss a part:
If you are 100% sure you miss a part, please contact sales@pollen-robotics.com with a picture of all the parts you have and order number or invoice number.  
You can also find [stl files](https://github.com/pollen-robotics/reachy_mini/tree/develop/src/reachy_mini/descriptions/reachy_mini/mjcf/assets) to print it by yourself in the meantime.
</details>


<details>
<summary><strong>Can't connect to my Wireless Reachy Mini using a USB-C cable</strong></summary>

Wireless units do not expose the robot over USB the way the Lite version does, so plugging a USB-C cable into your laptop will not give you a working connection.  
Instead:

- Join the robot to your Wi-Fi network and use the SDK client on your laptop to control it remotely.
- If you want to run code directly on the embedded Raspberry Pi, SSH in and execute your scripts there (this is what the Dashboard does after you publish/install an app).
- For a tethered link, use a USB-C-to-Ethernet adapter plus an Ethernet cable—this simply replaces Wi-Fi with wired Ethernet.

</details>

<details>
<summary><strong>Wireless Acces point doesn't show up - RPI doesn't boot</strong></summary>
There is a switch on the board in the head that needs to be in a given position. And if it's not, the AP doesn't show. It's possible that this switch was moved during assembly or maybe even a factory mistake.
Please check that the switch is on the "debug" and not on "download" position. See the picture below:

![switch_position](/docs/assets/wireless_switch.png)

</details>

#### If your issue/question is not listed here, please check the full FAQ below.

## 📋 FAQ Table of Contents

1.  [🚀 Getting Started & Assembly](#-getting-started--assembly)
2.  [🔌 Connection & Dashboard](#-connection--dashboard)
3.  [🤖 Hardware, Motors & Limits](#-hardware-motors--limits)
4.  [🐍 SDK, Apps & Programming](#-sdk-apps--programming)
5.  [🕹️ Moving the Robot](#-moving-the-robot)
6.  [👁️ Vision & Audio](#-vision--audio)
7.  [🔧 Specific Error Messages & Fixes](#-specific-error-messages--fixes)
8.  [📦 Shipping & Warranty](#-shipping--warranty)

<br>

## 🚀 Getting Started & Assembly

<details>
<summary><strong>How long does assembly usually take?</strong></summary>

Most testers report between **1.5 and 2 hours**. It can take up to 4 hours depending on your experience level.

</details>

<details>
<summary><strong>Are there any difficult steps during assembly?</strong></summary>

Not really, testers describe it as **fun, simple, and satisfying**. Basic tools and patience are enough. C**able routing** and **torqueing parts correctly** are the trickiest elements. When you buy a Reachy Mini Lite or Wireless, it comes with a printed user guide, and you also have access to a video and the Digital guide.
Video for Reachy Mini [BETA](https://www.youtube.com/watch?v=_r0cHySFbeY), LITE, WIRELESS

Digital Assembly Guide for Reachy Mini [BETA](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini_Assembly_Guide), LITE, WIRELESS

</details>

<details>
<summary><strong>I have 2 cables and a few screws left after finishing the assembly. Is this normal?</strong></summary>

Yes, this is completely normal.  
We intentionally include spare cables and screws in the kit in case some parts are damaged or lost during assembly.

You do not need to install them.

</details>

<details>
<summary><strong>My Reachy Mini doesn’t move on first startup. What should I check?</strong></summary>

* **Power Supply:** Ensure the 7V-5A power supply is plugged in. The USB connection is not enough to power the motors.
* **Cables:** Check that all cables are fully inserted. Loose power cables are a common cause of "motor not responding" errors.
* **Troubleshooting Section:** See the Essential Troubleshooting section at the top of this page.

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

See the [Reachy Mini Wireless guide](/docs/platforms/reachy_mini/get_started.md) for detailed instructions on connecting to Wi-Fi.

</details>

<details>
<summary><strong>How do I reset the Wi-Fi hotspot?</strong></summary>

If you need to reset the robot's Wi-Fi hotspot (for example, if you can't connect or want to change the network), follow the instructions in the [Wi-Fi Reset Guide](/docs/platforms/reachy_mini/reset.md).

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
3.  **Daemon:** Make sure the daemon `reachy-mini-daemon` is running in a terminal.

</details>

<details>
<summary><strong>Does the robot have a Web API?</strong></summary>

Yes. The daemon provides a REST API (FastAPI) and WebSocket support.
* **Docs:** `http://localhost:8000/docs` (available when daemon is running).
* **Features:** Get state, Move joints, Control daemon.

You can use the API to control the robot and get its state and even control the daemon itself. The API is implemented using [FastAPI](https://fastapi.tiangolo.com/) and [pydantic](https://docs.pydantic.dev/latest/) models.

It should provide you all the necessary endpoints to interact with the robot, including:

- Getting the state of the robot (joints positions, motor status, etc.)
- Moving the robot's joints or setting specific poses

The API is documented using OpenAPI, and you can access all available routes and test them at http://localhost:8000/docs when the daemon is running. You can also access the raw OpenAPI schema at http://localhost:8000/openapi.json.

This can be useful if you want to generate client code for your preferred programming language or framework, connect it to your AI application, or even to create your MCP server.

**WebSocket support**

The API also supports WebSocket connections for real-time updates. For instance, you can subscribe to joint state updates:

```
let ws = new WebSocket(`ws://127.0.0.1:8000/api/state/ws/full`);

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data);
};
```

</details>

<details>
<summary><strong>Why do I need a virtual environment (.venv)?</strong></summary>

Helps prevent package conflicts during SDK installation.

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
* If the motor's led blinks red, see the "Motor blinking red or Overload Error" section in the Essential Troubleshooting above.

</details>

<details>
<summary><strong>Does the battery has safety features?</strong></summary>
Wireless includes a proper battery charger.  
The battery integrates a BMS with a temperature sensor too.

</details>

<details>
<summary><strong>How do I see the battery left?</strong></summary>
We do not have the possibility to check the battery status, that's a known limitation of the design.  

We only have the led indication for "low battery" when it's time to charge it. (green -> orange -> red)

</details>

<details>
<summary><strong>How to remove the battery</strong></summary>

- Check that the green led is not on first.
- Remove the 3x screws at the bottom and take out the foot a little bit.
![remove_foot](/docs/assets/remove_foot.png)
- Unplug the indicated connector (red arrow) to be able to remove the battery. There should be some double-sided tape that maintain the battery in place, so it can be a bit hard to remove.
![battery_location](/docs/assets/battery_connector.png)
- When you'll re-assemble it, do these step again in reverse order. Just be careful not to pinch any cable.
</details>

<details>
<summary><strong>The head may touch the body during some official motions</strong></summary>

This behavior is expected and not a hardware or software bug.  
However, since it can be confusing, we will update those motions to avoid this contact.

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

<details><summary><strong>Is installing apps directly from the dashboard supported?</strong></summary>

Sure! You can install apps directly from your Dashboard if they’re native, or add them to your favourites if they’re web-based.

</details>

<details>
<summary><strong>All apps installations fail on Windows !</strong></summary>

It might be related to unsufficient rights to create symlinks in Windows. You can set the environment variable `HF_HUB_DISABLE_SYMLINKS_WARNING` to 1 to remove the warnings that cause the failure.

In a terminal, run :
```powershell
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

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

<details>
<summary><strong>How do I debug an app on the Wireless?</strong></summary>

SSH into the embedded computer, clone (or copy) your app, and run it manually. This reproduces what the dashboard does when launching your app.

```bash
ssh pollen@reachy-mini.local
# password: root
cd your_app_name
python your_app_name/main.py
```

Your GUI will open at the usual address (for example, `http://reachy-mini.local:8042`).

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
* Check that the flat flexible cable is intalled the right way (Slides 45-47 of assembly guide).

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
