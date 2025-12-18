# Troubleshooting & FAQ

Welcome to the Reachy Mini support page. Click on the questions below to reveal the answers.

## 📋 Table of Contents

1.  [🚀 Getting Started & Assembly](#-getting-started--assembly)
2.  [🐍 SDK, Apps & Programming](#-sdk-apps--programming)
3.  [🤖 Hardware & Customization](#-hardware--cutomization)
4.  [👁️ Media & Sensors](#-media--sensors)
5.  [👀 Interaction features](#-interactions-features)
7.  [🤝 Contributing](#-contributing)
8.  [📦 Shipping & Warranty](#-shipping--warranty)

---

## 1. 🚀 Getting Started & Assembly

### 1.1 Assembly

<details>
<summary>
<strong>How long does assembly usually take?</strong>
<br>Tags: <kbd>ASSEMBLY</kbd> <kbd>HARDWARE</kbd>
</summary>

Most testers report <b>1.5–2 hours</b>, with some up to <b>4 hours</b> depending on experience.

</details><br>


<details>
<summary>
<strong>Are there any difficult steps during assembly?</strong>
<br>Tags: <kbd>ASSEMBLY</kbd> <kbd>HARDWARE</kbd> <kbd>BETA</kbd> <kbd>LITE</kbd> <kbd>WIRELESS</kbd>
</summary>

Not really, testers describe it as **fun, simple, and satisfying**. Basic tools and patience are enough. 
**Cable routing** and **torqueing parts correctly** are the trickiest elements. When you buy a Reachy Mini Lite or Wireless, 
it comes with a printed user guide, and you also have access to a video and the Digital guide.
Video for Reachy Mini [BETA](https://www.youtube.com/watch?v=_r0cHySFbeY), LITE, WIRELESS

Digital Assembly Guide for Reachy Mini [BETA](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini_Assembly_Guide), LITE, WIRELESS

</details><br>

<details>
<summary>
<strong>How to remove the battery?</strong>
<br>Tags: <kbd>ASSEMBLY</kbd> <kbd>HARDWARE</kbd> <kbd>WIRELESS</kbd>
</summary>

- Check that the green led is not on first.
- Remove the 3x screws at the bottom and take out the foot a little bit.

![Bottom screws](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/bottom_raws.png)

- Unplug the indicated connector (red arrow) to be able to remove the battery. There should be some double-sided tape that maintain the battery in place, so it can be a bit hard to remove.

![Battery connector](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/battery_connector.png)

- When you'll re-assemble it, do these step again in reverse order. Just be careful not to pinch any cable.

</details><br>


### 1.2 Start

<details>
<summary>
<strong>My Reachy Mini doesn’t move on first startup. What should I check?</strong>
<br>Tags: <kbd>START</kbd> <kbd>POWER</kbd> <kbd>TROUBLESHOOTING</kbd> <kbd>BETA</kbd> <kbd>LITE</kbd> <kbd>WIRELESS</kbd>
</summary>

Verify the **power supply**, and ensure all cables are fully inserted, loose power cables caused several “motor not responding” issues.

</details><br>

<details>
<summary>
<strong>One antenna is reversed, how to fix this?</strong>
<br>Tags: <kbd>START</kbd> <kbd>TROUBLESHOOTING</kbd> <kbd>BETA</kbd> <kbd>LITE</kbd> <kbd>WIRELESS</kbd>
</summary>

If your robot’s antennas are positioned in an antagonistic way, as shown in the photo below, it means your robot is part of a batch that had a small production issue: the antennas were mounted the wrong way around.

![Antenna reversed](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/antenna_reversed.png)

**Here is a step-by-step guide to help you install them correctly**

|Step                          |                                   |
|------------------------------|-----------------------------------|
|![Antenna reversed step1](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/antenna-reversed-step1.png) | <p style="display: table-cell; vertical-align: middle; margin: 0;">STEP 1: Unscrew the antenna that has the issue, the one that points upward when the robot is in the OFF position and downward when it is in the ON position</p>|
|![Antenna reversed step2](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/antenna-reversed-step2.png) | <p style="display: table-cell; vertical-align: middle; margin: 0;"> STEP 2: Unscrew the two screws holding the antenna mount attached to the motor horn.</p> |
|![Antenna reversed step3](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/antenna-reversed-step3.png)|<p style="display: table-cell; vertical-align: middle; margin: 0;">STEP 3: Position the motor so that the **dot** (**A**) located next to the motor shaft **hole** (**B**) is facing upward</p>|
|![Antenna reversed step4](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/antenna-reversed-step4.png)|<p style="display: table-cell; vertical-align: middle; margin: 0;">STEP 4: Align the antenna interface so that the line mark (**2**) is also facing upward like the motor part (**1**).</p>|
|![Antenna reversed step4bis](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/antenna-reversed-step4bis.png)| <p style="display: table-cell; vertical-align: middle; margin: 0;">Screw the two 2.5×6 screws to attach the antenna interface to the motor horn.</p>|
|![Antenna reversed step5](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/antenna-reversed-step5.png)|<p style="display: table-cell; vertical-align: middle; margin: 0;">STEP 5: Fix back the Antenna with **2.5x8mm** screw.</p>|

It’s fixed! You can now enjoy all the features of Reachy Mini again. Head over to hf.co/reachy-mini to continue.
![Antenna ok](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/antenna_ok.png)


</details><br>


### 1.3 Set up

<details>
<summary>
<strong>How to Connect to a Wi-Fi network?</strong>
<br>Tags: <kbd>WiFi</kbd> <kbd>START</kbd> <kbd>WIRELESS</kbd>
</summary>

1. Power on your Reachy Mini.
2. Reachy Mini will create its own access point: "reachy-mini-ap". It should appear in the list of available Wi-Fi networks on your computer or smartphone after a few moments.
3. Connect your computer to the `reachy-mini-ap` Wi-Fi network (password: `reachy-mini`). Or you can directly scan the QR-code below to join the network:
    
    ![QR-Code reachy-mini-ap](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/qrcode-ap.png)
    
4. Open a web browser and go to http://reachy-mini.local:8000/settings to access the configuration page.
5. Enter your Wi-Fi network credentials (SSID and password) and click "Connect".
6. Wait a few moments for Reachy Mini to connect to your Wi-Fi network. The access point will disappear once connected. If the connection fails, Reachy Mini will restart the access point, and you can try again.

</details><br>


<details>
<summary>
<strong>Why do I need a virtual environment (.venv)?</strong>
<br>Tags: <kbd>SDK</kbd> <kbd>PYTHON</kbd> <kbd>ENVIRONMENT</kbd>
</summary>

Virtual environments keeps your Reachy Mini installation isolated and prevents conflicts with other Python projects. Modern Python development requires this now!

</details><br>


<details>
<summary>
<strong>How to solve Chromadb / GStreamer / OpenCV installation issues?</strong>
<br>Tags: <kbd>AUDIO</kbd> <kbd>VISION</kbd> <kbd>DEPENDENCIES</kbd>
</summary>

Users solved these by rebuilding their environment and reinstalling missing system dependencies.

</details><br>

<details>
<summary>
<strong>Do I need to start the daemon manually?</strong>
<br>Tags: <kbd>AUDIO</kbd> <kbd>VISION</kbd> <kbd>DEPENDENCIES</kbd>
</summary>

**NO** 

- With Reachy Mini (Wireless), the daemon is already running on the embedded Raspberry Pi.
- With Reachy Mini Lite, you can use [the desktop app](./reachy_mini_lite/get_started_lite.md).

</details><br>


### 1.4 Dashboard

<details>
<summary>
<strong>The dashboard on localhost:8000 doesn’t open. What am I missing?</strong>
<br>Tags: <kbd>DASHBOARD</kbd> <kbd>NETWORK</kbd> <kbd>SDK</kbd> <kbd>TROUBLESHOOTING</kbd>
</summary>

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

</details><br>

<details>
<summary>
<strong>Is installing apps directly from the dashboard supported?</strong>
<br>Tags: <kbd>DASHBOARD</kbd> <kbd>APPS</kbd> <kbd>INSTALLATION</kbd>
</summary>

Sure! You can install apps directly from your Dashboard if they’re native, or add them to your favourites if they’re web-based.

</details><br>


### 1.5 Tutorials / examples


<details>
<summary>
<strong>Is there a Scratch-like beginner mode?</strong>
<br>Tags: <kbd>TUTORIALS</kbd> <kbd>EDUCATION</kbd>
</summary>

One of our beta tester built a TurboWarp/Scratch 3.0 extension for controlling Reachy Mini: 
[reachy_mini_turbowarp](https://github.com/iizukak/reachy_mini_turbowarp)  

There is also a coding lab app using natural language commands to program movements on Reachy Mini:
[Reachy Mini Coding Lab](https://huggingface.co/spaces/dlouapre/coding_lab)

</details><br>


<details>
<summary>
<strong>Import errors when running examples.</strong>
<br>Tags: <kbd>SDK</kbd> <kbd>INSTALLATION</kbd>
</summary>

Outdated local SDK version — fixed by upgrading:
- With pip, run :
```bash
pip install -U reachy-mini
```
- With uv, run :
```bash
uv add reachy-mini
```

</details><br>



## 2. 🐍 SDK, Apps & Programming

### 2.1 Install Applications

<details>
<summary>
<strong>Installing an app from HuggingFace Space results in errors. What can I try?</strong>
<br>Tags: <kbd>INSTALLATION</kbd> <kbd>APPS</kbd> <kbd>SDK</kbd>
</summary>

Update your SDK. Early versions had a bug with Space installation.

```bash
pip install -U reachy-mini
```

</details><br>


<details>
<summary>
<strong>Where can I see examples of apps?</strong>
<br>Tags: <kbd>APPS</kbd> <kbd>EXAMPLES</kbd>
</summary>

Browse [spaces on Hugging Face](https://huggingface.co/spaces?q=reachy+mini) to discover all the apps developed for Reachy Mini. You can also find them directly through the Reachy Mini dashboard. The ones marked with a “certified” tag are those that have been tested and approved by the team.

</details><br>


<details>
<summary>
<strong>All apps installations fail on Windows !</strong>
<br>Tags: <kbd>APPS</kbd> <kbd>EXAMPLES</kbd>
</summary>

It might be related to unsufficient rights to create symlinks in Windows. You can set the environment variable `HF_HUB_DISABLE_SYMLINKS_WARNING` to 1 to remove the warnings that cause the failure.

In a terminal, run :
```bash
powershell set HF_HUB_DISABLE_SYMLINKS_WARNING=1 
```

</details><br>

### 2.2 Web API

<details>
<summary>
<strong>Does the robot have a Web API?</strong>
<br>Tags: <kbd>APPS</kbd> <kbd>SDK</kbd> <kbd>PROGRAMMING</kbd>
</summary>

The Reachy Mini daemon provides a REST API that you can use to control the robot and get its state and even control the daemon itself. The API is implemented using [FastAPI](https://fastapi.tiangolo.com/) and [pydantic](https://docs.pydantic.dev/latest/) models.

It should provide you all the necessary endpoints to interact with the robot, including:

- Getting the state of the robot (joints positions, motor status, etc.)
- Moving the robot's joints or setting specific poses

The API is documented using OpenAPI, and you can access all available routes and test them at `http://localhost:8000/docs` when the daemon is running. You can also access the raw OpenAPI schema at `http://localhost:8000/openapi.json`.

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

</details><br>


### 2.3 Develop Applications

<details>
<summary>
<strong>How do I write a Reachy Mini app?</strong>
<br>Tags: <kbd>APPS</kbd> <kbd>SDK</kbd> <kbd>PROGRAMMING</kbd>
</summary>

Inherit from `ReachyMiniApp` and implement `run`:

```python
import threading
from reachy_mini.apps.app import ReachyMiniApp
from reachy_mini import ReachyMini

class MyApp(ReachyMiniApp):
    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        # your app logic
        ...

```

- `stop_event` is a threading.Event indicating when the app should stop.

</details><br>


<details>
<summary>
<strong>How can I generate a new app template quickly?</strong>
<br>Tags: <kbd>APPS</kbd> <kbd>TOOLING</kbd>
</summary>

Use the app template generator:

```bash
reachy-mini-make-app my_app_name
```

This creates:

```
my_app_name/
├── pyproject.toml
├── README.md
├── my_app_name/
│   ├── __init__.py
│   └── main.py
```

Run it directly:

```bash
python my_app_name/main.py
```

Or install as a package:

```bash
pip install -e my_app_name/
```

</details><br>


<details>
<summary>
<strong>How do I add custom dance moves?</strong>
<br>Tags: <kbd>APPS</kbd> <kbd>MOVEMENT</kbd> <kbd>ANIMATION</kbd>
</summary>

Add sequences in `reachy_mini/app/collection/dance.py`.

</details><br>


<details>
<summary>
<strong>Is it possible to make Reachy nod to music (tempo sync)?</strong>
<br>Tags: <kbd>AUDIO</kbd> <kbd>MOVEMENT</kbd> <kbd>AI</kbd>
</summary>

Yes, but requires custom audio processing (e.g., Librosa). Streaming audio tempo detection is more challenging than offline analysis.

</details><br>


<details>
<summary>
<strong>Does the SDK support local AI actions (OpenAI-style)?</strong>
<br>Tags: <kbd>AI</kbd> <kbd>SDK</kbd> <kbd>APPS</kbd>
</summary>

Yes, users built local OpenAI-compatible integrations.

</details><br>


### 2.4 Simulation


<details>
<summary>
<strong>Is there a simulation environment?</strong>
<br>Tags: <kbd>SIMULATION</kbd> <kbd>SDK</kbd>
</summary>

**Yes**, via **MuJoCo**. It is still a work in progress, but you can run code with the `--sim` flag or `ReachyMini(media_backend="no_media")` if just testing logic without physics.

</details><br>

<details>
<summary>
<strong>Warning: "Circular buffer overrun"</strong>
<br>Tags: <kbd>SIMULATION</kbd> <kbd>TROUBLESHOOTING</kbd>
</summary>

When starting a client with `with ReachyMini() as mini:` in Mujoco (--sim mode), you may see the following warning:

```
Circular buffer overrun. To avoid, increase fifo_size URL option. To survive in such case, use overrun_nonfatal option
```

This message comes from FFmpeg (embedded in OpenCV) while consuming the UDP video stream. It appears because the frames are not being used, causing the buffer to fill up. If you do not intend to use the frames, set `ReachyMini(media_backend="no_media")` or `ReachyMini(media_backend="default_no_video")`.

</details><br>



### 2.5 Moving Reachy Mini


<details>
<summary>
<strong>How do I move Reachy Mini’s head to a specific pose?</strong>
<br>Tags: <kbd>MOVEMENT</kbd> <kbd>HEAD</kbd> <kbd>SDK</kbd>
</summary>

Use `goto_target` with a pose created by `create_head_pose`:

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as mini:
    mini.goto_target(head=create_head_pose(y=-10, mm=True))

```

- `create_head_pose` builds a 4x4 transform matrix (position + orientation).
- `mm=True` means translation arguments are in millimeters.
- The head frame is located at the base of the head.

</details><br>


<details>
<summary>
<strong>How do I control head orientation (roll, pitch, yaw)?</strong>
<br>Tags: <kbd>MOVEMENT</kbd> <kbd>HEAD</kbd>
</summary>

You can add orientation arguments to `create_head_pose`, for example:

```python
pose = create_head_pose(z=10, roll=15, degrees=True, mm=True)
mini.goto_target(head=pose, duration=2.0)

```

- `degrees=True` means angles are given in degrees.
- You can combine translation (x, y, z) and orientation (roll, pitch, yaw).

</details><br>


<details>
<summary>
<strong>How do I move head, body, and antennas at the same time?</strong>
<br>Tags: <kbd>MOVEMENT</kbd> <kbd>HEAD</kbd> <kbd>BODY</kbd> <kbd>ANTENNAS</kbd>
</summary>

Use `goto_target` with multiple named arguments:

```python
import numpy as np
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as mini:
    mini.goto_target(
        head=create_head_pose(y=-10, mm=True),
        antennas=np.deg2rad([45, 45]),
        duration=2.0,
        body_yaw=np.deg2rad(30),
    )

```

- `antennas` is a 2-element array in radians [right, left].
- `body_yaw` controls body rotation.

</details><br>


<details>
<summary>
<strong>What’s the difference between `goto_target` and `set_target`?</strong>
<br>Tags: <kbd>MOVEMENT</kbd> <kbd>CONTROL</kbd> <kbd>CONTROL</kbd>
</summary>

`goto_target`:

- Interpolates motion over a duration (default 0.5 s).
- Supports methods: `linear`, `minjerk`, `ease`, `cartoon`.
- Ideal for smooth, timed motions.

`set_target`:

- Sets the target immediately, without interpolation.
- Suited for high-frequency control (e.g. sinusoidal trajectories, teleoperation).

Example of sinusoidal motion:

```python
y = 10 * np.sin(2 * np.pi * 0.5 * t)
mini.set_target(head=create_head_pose(y=y, mm=True))

```

</details><br>


<details>
<summary>
<strong>How do I choose the interpolation method for movements?</strong>
<br>Tags: <kbd>MOVEMENT</kbd> <kbd>INTERPOLATION</kbd>
</summary>

Use the `method` argument in `goto_target`:

```python
mini.goto_target(
    head=create_head_pose(y=10, mm=True),
    antennas=np.deg2rad([-45, -45]),
    duration=2.0,
    method="cartoon",  # "linear", "minjerk", "ease", or "cartoon"
)

```

To compare methods, run the example:

- `examples/goto_interpolation_playground.py`

</details><br>


<details>
<summary>
<strong>How do I play predefined moves?</strong>
<br>Tags: <kbd>MOVEMENT</kbd> <kbd>DATASET</kbd> <kbd>PLAYBACK</kbd>
</summary>

Use `RecordedMoves` and `play_move`:

```python
from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves

with ReachyMini() as mini:
    recorded_moves = RecordedMoves("pollen-robotics/reachy-mini-dances-library")
    print(recorded_moves.list_moves())

    for move_name in recorded_moves.list_moves():
        print(f"Playing move: {move_name}")
        mini.play_move(recorded_moves.get(move_name), initial_goto_duration=1.0)
```

- `initial_goto_duration` smoothly moves the robot to the starting pose of the move.
- Datasets are hosted on Hugging Face (e.g. emotions / dances libraries).

</details><br>


<details>
<summary>
<strong>How do I record my own moves for later replay?</strong>
<br>Tags: <kbd>MOVEMENT</kbd> <kbd>RECORDING</kbd> <kbd>TELEOP</kbd>
</summary>

Call `start_recording()` and `stop_recording()` around the time where you send `set_target` commands:

```python
with ReachyMini() as mini:
    mini.start_recording()
    # run your teleop / control code ...
    recorded_motion = mini.stop_recording()
```

Each recorded frame contains:

- `time`
- `head`
- `antennas`
- `body_yaw`

Tools to record and upload datasets:

- `reachy_mini_toolbox/tools/moves`

</details><br>



## 3. 🤖 Hardware & Customization

### 3.1 Motors & Limits

<details>
<summary>
<strong>What are the safety limits of the head and body?</strong>
<br>Tags: <kbd>MOTORS</kbd> <kbd>LIMITS</kbd> <kbd>SAFETY</kbd>
</summary>

Limits:

1. Motors have limited mechanical range.
2. Head can collide with the body.
3. Body yaw: [-180°, 180°].
4. Head pitch and roll: [-40°, 40°].
5. Head yaw: [-180°, 180°].
6. Difference (body_yaw - head_yaw): [-65°, 65°].

If commanded pose exceeds these limits, the robot will clamp to the nearest safe pose (no exception is thrown).

</details><br>


<details>
<summary>
<strong>What happens if I ask for a pose that exceeds limits?</strong>
<br>Tags: <kbd>MOTORS</kbd> <kbd>LIMITS</kbd>
</summary>

Example:

```python
reachy.goto_target(head=create_head_pose(roll=-50, degrees=True))
```

- This exceeds the roll limit (±40°).
- The robot will move to the closest valid pose instead.

You can inspect the actual pose with:
```python
head_pose = reachy.get_current_head_pose()
print("current head pose", head_pose)
```

</details><br>


<details>
<summary>
<strong>What are the specs of the power supply?</strong>
<br>Tags: <kbd>POWER</kbd> <kbd>HARDWARE</kbd>
</summary>

Users requested this information; no final spec was posted in chatlogs.

</details><br>


<details>
<summary>
<strong>How do I reset motors?</strong>
<br>Tags: <kbd>MOTORS</kbd> <kbd>RESET</kbd>
</summary>

This is now done automatically when the daemon starts, so restarting the daemon should be enough.

</details><br>


<details>
<summary>
<strong>How do I enable, disable, or make motors compliant?</strong>
<br>Tags: <kbd>MOTORS</kbd> <kbd>COMPLIANCY</kbd> <kbd>SAFETY</kbd>
</summary>

- `enable_motors`
    
    Motors ON, robot holds its pose, you cannot move it by hand.
    
- `disable_motors`
    
    Motors OFF, robot is limp, you can move it freely by hand.
    
- `make_motors_compliant`
    
    Motors ON but compliant. Robot feels soft, does not resist; good for teaching by demonstration.
    
    Used by the gravity compensation example.

</details><br>


<details>
<summary>
<strong>Motors stop responding after a while.</strong>
<br>Tags: <kbd>MOTORS</kbd> <kbd>POWER</kbd> <kbd>TROUBLESHOOTING</kbd>
</summary>

Check power supply and cable stability. Updating SDK also solved issues for multiple users.

</details><br>


<details>
<summary>
<strong>Two motors side by side blink red</strong>
<br>Tags: <kbd>MOTORS</kbd> <kbd>ASSEMBLY</kbd> <kbd>HARDWARE</kbd> <kbd>WIRELESS</kbd> <kbd>LITE</kbd>
</summary>

After starting the robot, each motors should be in its neutral position:

![Neutral position](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/neutral_position.png)

If two motors side by side are blinking red, and are in a higher position (like motors 1&2 in the following picture), they are probably inverted.

![Motors 1 & 2](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/motors_1_2.png)

Fix: verify that the numbers on the motors match to the correct location and invert two motors if needed.

</details><br>


### 3.2 Customization


<details>
<summary>
<strong>Are CAD files available to customize Reachy Mini?</strong>
<br>Tags: <kbd>CAD</kbd> <kbd>3D_PRINTING</kbd>
</summary>

Not currently public.

</details><br>


<details>
<summary>
<strong>Can I modify the appearance (custom skins)?</strong>
<br>Tags: <kbd>CUSTOMIZATION</kbd> <kbd>SKINS</kbd>
</summary>

Community members shared custom builds, including a Star Wars astromech variant.

</details><br>




## 4. 👁️ Media & Sensors

### 4.1 Audio

<details>
<summary>
<strong>The audio-based app behaves inconsistently.</strong>
<br>Tags: <kbd>AUDIO</kbd> <kbd>REALTIME</kbd> <kbd>TROUBLESHOOTING</kbd>
</summary>

Streaming audio detection is harder than analyzing a complete file. Limitations come from real-time tempo extraction.

</details><br>

<details>
<summary>
<strong>No Microphone Input <i>(For beta only)</i></strong>
<br>Tags: <kbd>AUDIO</kbd> <kbd>MEDIA</kbd>
</summary>

There is a known issue where the microphone may not initialize correctly. Please update to [firmware 2.1.3](https://github.com/pollen-robotics/reachy_mini/blob/develop/src/reachy_mini/assets/firmware/reachymini_ua_io16_lin_v2.1.3.bin). You may need to run the [update script](https://github.com/pollen-robotics/reachy_mini/blob/develop/src/reachy_mini/assets/firmware/update.sh). Linux users may require to run the command as *sudo*.

Afterwards, run [examples/debug/sound_record.py](https://github.com/pollen-robotics/reachy_mini/blob/develop/examples/debug/sound_record.py) to check that everything is working properly.

If the problem persists, check the connection of the flex cables ([see slides 45 to 47](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini_Assembly_Guide)).

</details><br>

<details>
<summary>
<strong>Sound Direction of Arrival Not Working <i>(For beta only)</i></strong>
<br>Tags: <kbd>AUDIO</kbd> <kbd>MEDIA</kbd>
</summary>

The microphone array requires firmware version 2.1.0 or higher to support this feature. The firmware files are located in `src/reachy_mini/assets/firmware/*.bin`.

A [helper script](https://github.com/pollen-robotics/reachy_mini/blob/develop/src/reachy_mini/assets/firmware/update.sh) is available for Unix users (see above). Refer to the [Seeed documentation](https://wiki.seeedstudio.com/respeaker_xvf3800_introduction/#update-firmware) for more details on the upgrade process.

</details><br>

<details>
<summary>
<strong>Volume Is Too Low <i>(Linux only)</i></strong>
<br>Tags: <kbd>AUDIO</kbd> <kbd>MEDIA</kbd>
</summary>

1.  Run `alsamixer`.
2.  Set **PCM1** to 100%.
3.  Use **PCM,0** to adjust the global volume.

To make it permanent:
```bash
CARD=$(aplay -l | grep -i "reSpeaker" | head -n1 | sed -n 's/^card \([0-9]*\):.*/\1/p')
amixer -c "$CARD" set PCM,1 100%
sudo alsactl store "$CARD"
```

</details><br>

<details>
<summary>
<strong>How do I access microphone audio samples?</strong>
<br>Tags: <kbd>AUDIO</kbd> <kbd>MEDIA</kbd>
</summary>

```python
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    sample = mini.media.get_audio_sample()
    # sample is a numpy array as returned by sounddevice
```

</details><br>


<details>
<summary>
<strong>How do I send audio to the speaker?</strong>
<br>Tags: <kbd>AUDIO</kbd> <kbd>MEDIA</kbd>
</summary>

```python
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    # chunk is a numpy array of audio samples
    mini.media.push_audio_sample(chunk)
```

</details><br>

### 4.2 Vision


<details>
<summary>
<strong>Is camera quality good in various lighting conditions?</strong>
<br>Tags: <kbd>VISION</kbd> <kbd>CAMERA</kbd> <kbd>HARDWARE</kbd>
</summary>

Several testers reported excellent indoor performance, even in office lighting.

</details><br>


<details>
<summary>
<strong>How do I grab camera frames from Reachy Mini?</strong>
<br>Tags: <kbd>VISION</kbd> <kbd>CAMERA</kbd> <kbd>MEDIA</kbd>
</summary>

Use the media object:
```python
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    frame = mini.media.get_frame()
    # frame is a numpy array compatible with OpenCV
```

</details><br>


<details>
<summary>
<strong>How can I use the GStreamer backend instead of default OpenCV/sounddevice?</strong>
<br>Tags: <kbd>AUDIO</kbd> <kbd>VIDEO</kbd> <kbd>GSTREAMER</kbd> <kbd>ADVANCED</kbd>
</summary>

Install with the GStreamer extra:

```bash
pip install -e ".[gstreamer]"
```

Then run your code with `--backend gstreamer`.

You must have GStreamer binaries installed on your system. You can define custom pipelines (see `camera_gstreamer.py` in the repository for an example).

</details><br>



## 6. 👀 Interaction feature

<details>
<summary>
<strong>Can Reachy wake up or move autonomously based on audio?</strong>
<br>Tags: <kbd>AUDIO</kbd> <kbd>MOVEMENT</kbd> <kbd>AI</kbd>
</summary>

Yes, users implemented wake-up behaviors and audio-reactive motions in the Radio App.

</details><br>


<details>
<summary>
<strong>Can Reachy follow faces?</strong>
<br>Tags: <kbd>VISION</kbd> <kbd>FACE_TRACKING</kbd>
</summary>

Yes, with GStreamer + OpenCV, users achieved real-time face detection successfully.

</details><br>

<details>
<summary>
<strong>Reachy’s face tracking seems slow or inaccurate.</strong>
<br>Tags: <kbd>VISION</kbd> <kbd>FACE_TRACKING</kbd>
</summary>

Performance relies heavily on **lighting conditions**. Ensure the face is well-lit. Using the GStreamer backend can also improve latency compared to the default OpenCV backend.

</details><br>


<details>
<summary>
<strong>Can I create a personalized behavior model?</strong>
<br>Tags: <kbd>AI</kbd> <kbd>PERSONALIZATION</kbd>
</summary>

Some users suggested discriminator-based personalization (“Make Reachy yours”).

</details><br>


<details>
<summary>
<strong>How can I make Reachy Mini look at a point in the image?</strong>
<br>Tags: <kbd>VISION</kbd> <kbd>LOOK_AT_IMAGE</kbd>
</summary>

Use the `look_at_image` method (see `look_at_image.py` example).

You provide a 2D point in image coordinates:

- (0, 0) = top-left of the image
- (width, height) = bottom-right

You can also specify the duration of the movement, like in `goto_target`.

</details><br>


<details>
<summary>
<strong>How can I make Reachy Mini look at a 3D point in the world?</strong>
<br>Tags: <kbd>VISION</kbd> <kbd>LOOK_AT_WORLD</kbd>
</summary>

Use `look_at_world`, which takes a 3D point in the robot world frame.

The world frame is illustrated in the docs (world_frame image).

</details><br>



## 7. 🤝 Contributing


<details>
<summary>
<strong>Is there a recommended way to share apps with the community?</strong>
<br>Tags: <kbd>APPS</kbd> <kbd>COMMUNITY</kbd> <kbd>CONTRIBUTING</kbd>
</summary>

Recommended workflow:

- Wrap your behavior in a `ReachyMiniApp`.
- Publish as a Hugging Face Space (web based and added as a favourite on your dashboard) or Python package (directly installable via the dashboard).
- See the example space: [`reachy_mini_app_example`](https://huggingface.co/spaces/pollen-robotics/reachy_mini_app_example).

</details><br>


<details>
<summary>
<strong>How can I submit improvements to the SDK?</strong>
<br>Tags: <kbd>SDK</kbd> <kbd>CONTRIBUTING</kbd>
</summary>

Via pull requests on our [GitHub](https://github.com/pollen-robotics/reachy_mini/blob/develop/README.md).

</details><br>


<details>
<summary>
<strong>How can I contribute to datasets of moves (dances/emotions)?</strong>
<br>Tags: <kbd>DATASET</kbd> <kbd>CONTRIBUTING</kbd> <kbd>MOVEMENT</kbd>
</summary>

Use the tools in:

- `reachy_mini_toolbox/tools/moves`

They help you:

- Record moves (via `start_recording` / `stop_recording`).
- Upload them to the Hugging Face Hub (for example: `reachy-mini-emotions-library`, `reachy-mini-dances-library`).

</details><br>



## 8. 📦 Shipping & Warranty

<details>
<summary>
<strong>My package is damaged or missing.</strong>
<br>Tags: <kbd>SHIPPING</kbd>
</summary>

Contact **Pollen Robotics** team immediately. You can send us an email to sales@pollen-robotics.com with photos of the package, receipt number or invoice number and your full name. We will then check with the transport company and keep you updated.

</details><br>


<details>
<summary>
<strong>Refund Policy</strong>
<br>Tags: <kbd>REFUND</kbd>
</summary>

* **Before shipping:** Contact `sales@pollen-robotics.com` for a 100% refund.
* **After shipping:** You have 30 days to return your package. Contact sales (sales@pollen-robotics.com) with proof of delivery and invoice or receipt number. If you have comments / feedback, please let us know, our focus is building a robot the open-source community enjoys building. 

</details><br>


<details>
<summary>
<strong>Warranty: My unit is malfunctioning / broken part ?</strong>
<br>Tags: <kbd>WARRANTY</kbd>
</summary>

If a part is broken/malfunctioning, Pollen's after-sales team will determine if it is a hardware defect covered by warranty. Then, our manufacturer will provide repair or replacement parts. You can send us an email to sales@pollen-robotics.com with photos of the issue, receipt number or invoice number and your full name.

</details><br>



---

## 💬 Still stuck?

If you couldn't find the answer to your issue in this guide, please reach out to us directly!
The Pollen Robotics team and the community are active on Discord to help you troubleshoot specific problems.

👉 **[Join the Pollen Robotics Discord](https://discord.gg/2bAhWfXme9)**