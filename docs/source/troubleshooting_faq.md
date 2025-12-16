<!-- FAQ blocks are automatically generated with ../../scripts/generate_fa.py.
Do not modify the content of the blocks manually.
If you want to modify or add a block, check ../README.md  -->

# Troubleshooting & FAQ

Welcome to the Reachy Mini support page. Click on the questions below to reveal the answers.

## 📋 Table of Contents

1.  [🚀 Getting Started & Assembly](#-getting-started--assembly)
2.  [🔌 Connection & Dashboard](#-connection--dashboard)
3.  [🤖 Hardware, Motors & Limits](#-hardware-motors--limits)
4.  [🐍 SDK, Apps & Programming](#-sdk-apps--programming)
5.  [🕹️ Moving the Robot](#-moving-the-robot)
6.  [👁️ Vision & Audio](#-vision--audio)
7.  [🔧 Specific Error Messages & Fixes](#-specific-error-messages--fixes)
8.  [📦 Shipping & Warranty](#-shipping--warranty)

<br>

## 1. 🚀 Getting Started & Assembly

### 1.1 Assembly

<!-- FAQ:1_getting_started:assembly:start -->

<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>How long does assembly usually take?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">ASSEMBLY</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">HARDWARE</kbd></summary>

Most testers report <b>1.5–2 hours</b>, with some up to <b>4 hours</b> depending on experience.

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>Are there any difficult steps during assembly?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">ASSEMBLY</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">HARDWARE</kbd></summary>

Not really, testers describe it as **fun, simple, and satisfying**. Basic tools and patience are enough. 
**Cable routing** and **torqueing parts correctly** are the trickiest elements. When you buy a Reachy Mini Lite or Wireless, 
it comes with a printed user guide, and you also have access to a video and the Digital guide.
Video for Reachy Mini [BETA](https://www.youtube.com/watch?v=_r0cHySFbeY), LITE, WIRELESS

Digital Assembly Guide for Reachy Mini [BETA](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini_Assembly_Guide), LITE, WIRELESS

</details><br>


<!-- FAQ:1_getting_started:assembly:end -->

### 1.2 Set up

<!-- FAQ:1_getting_started:setup:start -->

<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>How to Connect to a Wi-Fi network?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">WiFi</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">START</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">WIRELESS</kbd></summary>

1. Power on your Reachy Mini.
2. Reachy Mini will create its own access point: "reachy-mini-ap". It should appear in the list of available Wi-Fi networks on your computer or smartphone after a few moments.
3. Connect your computer to the `reachy-mini-ap` Wi-Fi network (password: `reachy-mini`). Or you can directly scan the QR-code below to join the network:
    
    ![QR-Code reachy-mini-ap](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/qrcode-ap.png)
    
4. Open a web browser and go to http://reachy-mini.local:8000/settings to access the configuration page.
5. Enter your Wi-Fi network credentials (SSID and password) and click "Connect".
6. Wait a few moments for Reachy Mini to connect to your Wi-Fi network. The access point will disappear once connected. If the connection fails, Reachy Mini will restart the access point, and you can try again.

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>Why do I need a virtual environment (.venv)?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">SDK</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">PYTHON</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">ENVIRONMENT</kbd></summary>

Helps prevent package conflicts during SDK installation.

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>How to solve Chromadb / GStreamer / OpenCV installation issues?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">AUDIO</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">VISION</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">DEPENDENCIES</kbd></summary>

Users solved these by rebuilding their environment and reinstalling missing system dependencies.

</details><br>


<!-- FAQ:1_getting_started:setup:end -->

### 1.3 Dashboard

<!-- FAQ:1_getting_started:dashboard:start -->

<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>Is installing apps directly from the dashboard supported?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">DASHBOARD</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">APPS</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">INSTALLATION</kbd></summary>

Sure! You can install apps directly from your Dashboard if they’re native, or add them to your favourites if they’re web-based.

</details><br>


<!-- FAQ:1_getting_started:dashboard:end -->

### 1.4 Tutorials

<!-- FAQ:1_getting_started:tutorials:start -->

<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>Is there a Scratch-like beginner mode?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">TUTORIALS</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">EDUCATION</kbd></summary>

One of our beta tester built a TurboWarp/Scratch 3.0 extension for controlling Reachy Mini: 
[reachy_mini_turbowarp](https://github.com/iizukak/reachy_mini_turbowarp)

</details><br>


<!-- FAQ:1_getting_started:tutorials:end -->


## 🧩 2. Using and Developing applications / spaces

### 2.1 Install Applications

<!-- FAQ:2_using_applications_spaces:install_apps:start -->

<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>Installing an app from HuggingFace Space results in errors. What can I try?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">INSTALLATION</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">APPS</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">SDK</kbd></summary>

Update the Reachy Mini SDK to the latest version. Earlier versions had a bug preventing smooth installation.

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>Where can I see examples of apps?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">APPS</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">EXAMPLES</kbd></summary>

Browse [spaces on Hugging Face](https://huggingface.co/spaces?q=reachy+mini) to discover all the apps developed for Reachy Mini. You can also find them directly through the Reachy Mini dashboard. The ones marked with a “certified” tag are those that have been tested and approved by the team.

</details><br>


<!-- FAQ:2_using_applications_spaces:install_apps:end -->

### 2.2 Web API

### 2.3 Develop Applications

<!-- FAQ:2_using_applications_spaces:develop_apps:start -->

<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>How do I write a Reachy Mini app?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">APPS</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">SDK</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">PROGRAMMING</kbd></summary>

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


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>How can I generate a new app template quickly?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">APPS</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">TOOLING</kbd></summary>

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


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>How do I add custom dance moves?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">APPS</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">MOVEMENT</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">ANIMATION</kbd></summary>

Add sequences in `reachy_mini/app/collection/dance.py`.

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>Is it possible to make Reachy nod to music (tempo sync)?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">AUDIO</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">MOVEMENT</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">AI</kbd></summary>

Yes, but requires custom audio processing (e.g., Librosa). Streaming audio tempo detection is more challenging than offline analysis.

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>Does the SDK support local AI actions (OpenAI-style)?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">AI</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">SDK</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">APPS</kbd></summary>

Yes, users built local OpenAI-compatible integrations.

</details><br>


<!-- FAQ:2_using_applications_spaces:develop_apps:end -->

### 2.4 Simulation

<!-- FAQ:2_using_applications_spaces:simulation:start -->

<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>Is there a simulation environment?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">SIMULATION</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">SDK</kbd></summary>

Mentioned in discussions; models were being worked on but not yet official during beta.

</details><br>


<!-- FAQ:2_using_applications_spaces:simulation:end -->

### 2.5 Moving Reachy Mini

<!-- FAQ:2_using_applications_spaces:moving_robot:start -->

<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>How do I move Reachy Mini’s head to a specific pose?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">MOVEMENT</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">HEAD</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">SDK</kbd></summary>

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


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>How do I control head orientation (roll, pitch, yaw)?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">MOVEMENT</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">HEAD</kbd></summary>

You can add orientation arguments to `create_head_pose`, for example:

```python
pose = create_head_pose(z=10, roll=15, degrees=True, mm=True)
mini.goto_target(head=pose, duration=2.0)

```

- `degrees=True` means angles are given in degrees.
- You can combine translation (x, y, z) and orientation (roll, pitch, yaw).

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>How do I move head, body, and antennas at the same time?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">MOVEMENT</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">HEAD</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">BODY</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">ANTENNAS</kbd></summary>

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


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>What’s the difference between `goto_target` and `set_target`?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">MOVEMENT</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">CONTROL</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">CONTROL</kbd></summary>

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


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>How do I choose the interpolation method for movements?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">MOVEMENT</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">INTERPOLATION</kbd></summary>

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


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>How do I play predefined moves?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">MOVEMENT</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">DATASET</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">PLAYBACK</kbd></summary>

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


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>How do I record my own moves for later replay?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">MOVEMENT</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">RECORDING</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">TELEOP</kbd></summary>

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


<!-- FAQ:2_using_applications_spaces:moving_robot:end -->

## 🎛 3. Hardware Guide

<!-- FAQ:3_hardware_guide:hardware_guide:start -->

<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>What are the safety limits of the head and body?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">MOTORS</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">LIMITS</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">SAFETY</kbd></summary>

Limits:

1. Motors have limited mechanical range.
2. Head can collide with the body.
3. Body yaw: [-180°, 180°].
4. Head pitch and roll: [-40°, 40°].
5. Head yaw: [-180°, 180°].
6. Difference (body_yaw - head_yaw): [-65°, 65°].

If commanded pose exceeds these limits, the robot will clamp to the nearest safe pose (no exception is thrown).

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>What happens if I ask for a pose that exceeds limits?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">MOTORS</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">LIMITS</kbd></summary>

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


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>What are the specs of the power supply?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">POWER</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">HARDWARE</kbd></summary>

Users requested this information; no final spec was posted in chatlogs.

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>How do I reset motors?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">MOTORS</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">RESET</kbd></summary>

This is now done automatically when the daemon starts, so restarting the daemon should be enough.

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>How do I enable, disable, or make motors compliant?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">MOTORS</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">COMPLIANCY</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">SAFETY</kbd></summary>

- `enable_motors`
    
    Motors ON, robot holds its pose, you cannot move it by hand.
    
- `disable_motors`
    
    Motors OFF, robot is limp, you can move it freely by hand.
    
- `make_motors_compliant`
    
    Motors ON but compliant. Robot feels soft, does not resist; good for teaching by demonstration.
    
    Used by the gravity compensation example.

</details><br>


<!-- FAQ:3_hardware_guide:hardware_guide:end -->



## 🧠 4. Media & Sensors

### 4.1 Audio

<!-- FAQ:4_media_sensors:audio:start -->

<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>How do I access microphone audio samples?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">AUDIO</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">MEDIA</kbd></summary>

```python
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    sample = mini.media.get_audio_sample()
    # sample is a numpy array as returned by sounddevice
```

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>How do I send audio to the speaker?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">AUDIO</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">MEDIA</kbd></summary>

```python
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    # chunk is a numpy array of audio samples
    mini.media.push_audio_sample(chunk)
```

</details><br>


<!-- FAQ:4_media_sensors:audio:end -->


### 4.2 Vision

<!-- FAQ:4_media_sensors:vision:start -->

<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>Is camera quality good in various lighting conditions?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">VISION</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">CAMERA</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">HARDWARE</kbd></summary>

Several testers reported excellent indoor performance, even in office lighting.

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>How do I grab camera frames from Reachy Mini?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">VISION</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">CAMERA</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">MEDIA</kbd></summary>

Use the media object:
```python
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    frame = mini.media.get_frame()
    # frame is a numpy array compatible with OpenCV
```

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>How can I use the GStreamer backend instead of default OpenCV/sounddevice?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">AUDIO</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">VIDEO</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">GSTREAMER</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">ADVANCED</kbd></summary>

Install with the GStreamer extra:

```bash
pip install -e ".[gstreamer]"
```

Then run your code with `--backend gstreamer`.

You must have GStreamer binaries installed on your system. You can define custom pipelines (see `camera_gstreamer.py` in the repository for an example).

</details><br>


<!-- FAQ:4_media_sensors:vision:end -->

 

## 🎨 5. Customization

<!-- FAQ:5_customization:customization:start -->

<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>Are CAD files available to customize Reachy Mini?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">CAD</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">3D_PRINTING</kbd></summary>

Users asked for CAD files; no direct confirmation was provided in chatlogs.

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>Can I modify the appearance (custom skins)?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">CUSTOMIZATION</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">SKINS</kbd></summary>

Community members shared custom builds, including a Star Wars astromech variant.

</details><br>


<!-- FAQ:5_customization:customization:end -->



## 👀 6. Interaction feature

<!-- FAQ:6_interaction_features:interaction_features:start -->

<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>Can Reachy wake up or move autonomously based on audio?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">AUDIO</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">MOVEMENT</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">AI</kbd></summary>

Yes, users implemented wake-up behaviors and audio-reactive motions in the Radio App.

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>Can Reachy follow faces?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">VISION</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">FACE_TRACKING</kbd></summary>

Yes, with GStreamer + OpenCV, users achieved real-time face detection successfully.

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>Can I create a personalized behavior model?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">AI</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">PERSONALIZATION</kbd></summary>

Some users suggested discriminator-based personalization (“Make Reachy yours”).

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>How can I make Reachy Mini look at a point in the image?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">VISION</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">LOOK_AT_IMAGE</kbd></summary>

Use the `look_at_image` method (see `look_at_image.py` example).

You provide a 2D point in image coordinates:

- (0, 0) = top-left of the image
- (width, height) = bottom-right

You can also specify the duration of the movement, like in `goto_target`.

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>How can I make Reachy Mini look at a 3D point in the world?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">VISION</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">LOOK_AT_WORLD</kbd></summary>

Use `look_at_world`, which takes a 3D point in the robot world frame.

The world frame is illustrated in the docs (world_frame image).

</details><br>


<!-- FAQ:6_interaction_features:interaction_features:end -->



## 🤝 7. Contributing

<!-- FAQ:7_contributing:contributing:start -->

<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>Is there a recommended way to share apps with the community?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">APPS</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">COMMUNITY</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">CONTRIBUTING</kbd></summary>

Recommended workflow:

- Wrap your behavior in a `ReachyMiniApp`.
- Publish as a Hugging Face Space (web based and added as a favourite on your dashboard) or Python package (directly installable via the dashboard).
- See the example space: [`reachy_mini_app_example`](https://huggingface.co/spaces/pollen-robotics/reachy_mini_app_example).

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>How can I submit improvements to the SDK?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">SDK</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">CONTRIBUTING</kbd></summary>

Via pull requests on our [GitHub](https://github.com/pollen-robotics/reachy_mini/blob/develop/README.md).

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>How can I contribute to datasets of moves (dances/emotions)?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">CODING DATASET</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">CONTRIBUTING</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">MOVEMENT</kbd></summary>

Use the tools in:

- `reachy_mini_toolbox/tools/moves`

They help you:

- Record moves (via `start_recording` / `stop_recording`).
- Upload them to the Hugging Face Hub (for example: `reachy-mini-emotions-library`, `reachy-mini-dances-library`).

</details><br>


<!-- FAQ:7_contributing:contributing:end -->



## 📦 8. Post-sales

<!-- FAQ:8_post_sales:post_sales:start -->

<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>I haven’t received my package. How can I check it?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">DELIVERY</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">SHIPPING</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">POST_SALES</kbd></summary>

TODO

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>I received my package but its severely damage.</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">DELIVERY</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">SHIPPING</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">POST_SALES</kbd></summary>

If the damage is caused during shipment, we will claim with the transporter and then provide related compensation or solutions to the clients.

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>Can I have a refund (before the unit being shipped)?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">REFUND</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">POST_SALES</kbd></summary>

TODO

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>Can I have a refund (after the unit being shipped)?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">REFUND</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">POST_SALES</kbd></summary>

TODO

</details><br>


<details
  style="
    margin-top: 8px;
    border-left: 3px solid #3190d4;
    padding-left: 12px;
  "
>
<summary><strong>Warranty: My unit is malfunctioning / broken part ?</strong><br>Tags: <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">WARRANTY</kbd> <kbd style="display:inline-block;padding:2px 10px;margin:2px 4px;background:rgba(59,176,209,0.1);color:#3bb0d1;border-radius:12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;border:none;">POST_SALES</kbd></summary>

Our after-sale team will involve in to confirm if it's the hardware problem and if it's under warranty, if it is, we will either repair or replace the parts, depending on the situation.

</details><br>


<!-- FAQ:8_post_sales:post_sales:end -->


---

## 💬 Still stuck?

If you couldn't find the answer to your issue in this guide, please reach out to us directly!
The Pollen Robotics team and the community are active on Discord to help you troubleshoot specific problems.

👉 **[Join the Pollen Robotics Discord](https://discord.gg/2bAhWfXme9)**