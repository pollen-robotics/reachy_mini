# Reachy Mini 🤖

[![Ask on HuggingChat](https://img.shields.io/badge/Ask_on-HuggingChat-yellow?logo=huggingface&logoColor=yellow&style=for-the-badge)](https://huggingface.co/chat/?attachments=https%3A%2F%2Fgist.githubusercontent.com%2FFabienDanieau%2F919e1d7468fb16e70dbe984bdc277bba%2Fraw%2Fdoc_reachy_mini_full.md&prompt=Read%20this%20documentation%20about%20Reachy%20Mini%20so%20I%20can%20ask%20questions%20about%20it.)
[![Discord](https://img.shields.io/badge/Discord-Join_the_Community-7289DA?logo=discord&logoColor=white)](https://discord.gg/2bAhWfXme9)

**Reachy Mini is an open-source, expressive robot made for hackers and AI builders.**

It removes the hardware complexity, allowing you to focus on what matters: building intelligent agents that can see, hear, and interact with the physical world.

[![Reachy Mini Hello](/docs/assets/reachy_mini_hello.gif)](https://www.pollen-robotics.com/reachy-mini/)

## 🚀 Getting Started


Reachy Mini's hardware comes in two flavors:
- **Reachy Mini lite**: where the robot is directly connected to your computer via USB. And the code that controls the robot (the daemon) runs on your computer.
- **Reachy Mini wireless**: where an Raspberry Pi is embedded in the robot, and the code that controls the robot (the daemon) runs on the Raspberry Pi. You can connect to it via Wi-Fi from your computer (see [Wireless Setup](./docs/source/wireless-version.mdx)).

**Choose your platform to access the specific guide:**


| **🤖 Reachy Mini (Wireless)** | **🔌 Reachy Mini Lite** | **💻 Simulation** |
| :---: | :---: | :---: |
| The full autonomous experience.<br>Raspberry Pi 4 + Battery + WiFi. | The developer version.<br>USB connection to your computer. | No hardware required.<br>Prototype in MuJoCo. |
| [**👉 Go to Wireless Guide**](docs/platforms/reachy_mini/get_started.md) | [**👉 Go to Lite Guide**](docs/platforms/reachy_mini_lite/get_started.md) | [**👉 Go to Simulation**](docs/platforms/simulation/get_started.md) |

> ⚡ **Pro tip:** Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for 10-100x faster app installations (auto-detected, falls back to `pip`).

<br>

## 📱 Apps & Ecosystem

Reachy Mini comes with an app store powered by Hugging Face Spaces. You can install these apps directly from your robot's dashboard with one click!

* **🗣️ [Conversation App](https://huggingface.co/spaces/pollen-robotics/reachy_mini_conversation_app):** Talk naturally with Reachy Mini (powered by LLMs).
* **📻 [Radio](https://huggingface.co/spaces/pollen-robotics/reachy_mini_radio):** Listen to the radio with Reachy Mini !
* **👋 [Hand Tracker](https://huggingface.co/spaces/pollen-robotics/hand_tracker_v2):** The robot follows your hand movements in real-time.

[**👉 Browse all apps on Hugging Face**](https://pollen-robotics-reachy-mini-landing-page.hf.space/#/apps)

<br>

## 🐍 Software & SDK

Once your robot (or simulation) is running, the code is the same!
Control your Reachy Mini with Python to create movements, build apps, and connect AI models.

You will find the Installation guide, Quickstart, and API Reference in the SDK documentation.

[**👉 Go to SDK Documentation**](docs/SDK/readme.md)
[**🤗 Share your app with the community**](**https://huggingface.co/blog/pollen-robotics/make-and-publish-your-reachy-mini-apps**)

<br>

## 🛠 Hardware Overview

Reachy Mini robots are sold as kits and generally take **2 to 3 hours** to assemble. Detailed step-by-step guides are available in the platform-specific folders linked above.

* **Reachy Mini (Wireless):** Runs onboard (RPi 4), autonomous, includes IMU. [See specs](docs/platforms/reachy_mini/hardware.md).
* **Reachy Mini Lite:** Runs on your PC, powered via wall outlet. [See specs](docs/platforms/reachy_mini_lite/hardware.md).

[**🛒 Buy Reachy Mini**](https://www.hf.co/reachy-mini/)

<br>

## ❓ Troubleshooting

Encountering an issue? 👉 **[Check the Troubleshooting & FAQ Guide](/docs/troubleshooting.md)**

<br>

## 🤝 Community & Contributing

Reachy Mini is a collaborative project between [Pollen Robotics](https://www.pollen-robotics.com) and [Hugging Face](https://huggingface.co/).


```bash
pip install reachy-mini
```

From the source code, you can install the package with:

```bash
git clone https://github.com/pollen-robotics/reachy_mini
pip install -e ./reachy_mini
```

*Note that uv users can directly run the daemon with:*
```bash
uv run reachy-mini-daemon
```

The same package provides both the daemon and the Python SDK.

### Linux udev rules setup

On Linux systems, you need to set up udev rules to allow non-root access to the Reachy Mini hardware. Create the udev rules file with:

```bash
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", GROUP="dialout" #Reachy Mini
SUBSYSTEM=="tty", ATTRS{idVendor}=="38fb", ATTRS{idProduct}=="1001", MODE="0666", GROUP="dialout" #Reachy Mini soundcard' \
| sudo tee /etc/udev/rules.d/99-reachy-mini.rules
```

After saving the file, refresh the udev rules:

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Finally, add your current user to the `dialout` group:

```bash
sudo usermod -aG dialout $USER
```

You may need to log out and log back in for the group changes to take effect.

## Run the reachy mini daemon

Before being able to use the robot, you need to run the daemon that will handle the communication with the motors. This daemon can run either in simulation (MuJoCo) or on the real robot.

### For the real robot

```bash
reachy-mini-daemon
```

or run it via the Python module:

```bash
python -m reachy_mini.daemon.app.main
```

Additional argument for both simulation and real robot:

```bash
--localhost-only: (default behavior). The server will only accept connections from localhost.
```

or

```bash
--no-localhost-only: If set, the server will accept connections from any connection on the local network.
```

> [!TIP]
> **For the lite version (connected via USB)**
> 
> It should automatically detect the serial port of the robot. If it does not, you can specify it manually with the `-p` option:
> 
> ```bash
> reachy-mini-daemon -p <serial_port>
> ```

### In simulation ([MuJoCo](https://mujoco.org))

You first have to install the optional dependency `mujoco`.

```bash
pip install reachy-mini[mujoco]
```

Then run the daemon with the `--sim` argument.

```bash
reachy-mini-daemon --sim
```

Additional arguments:

```bash
--scene <empty|minimal> : (Default empty). Choose between a basic empty scene, or a scene with a table and some objects.
```

<img src="https://www.pollen-robotics.com/wp-content/uploads/2025/06/Reachy_mini_simulation.gif" width="250" alt="Reachy Mini in MuJoCo">


*Note: On OSX in order to run mujoco, you need to use mjpython (see [here](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)). So, you should run the daemon with:*

```bash
 mjpython -m reachy_mini.daemon.app.main --sim
 ```

### Usage

For more information about the daemon and its options, you can run:

```bash
reachy-mini-daemon --help
```

### Dashboard

You can access a simple dashboard to monitor the robot's status at [http://localhost:8000/](http://localhost:8000/) when the daemon is running. This lets you turn your robot on and off, run some basic movements, and browse spaces for Reachy Mini!

![Reachy Mini Dashboard](docs/assets/dashboard.png)

## Run the demo & awesome apps

Conversational demo for the Reachy Mini robot combining LLM realtime APIs, vision pipelines, and choreographed motion libraries: [reachy_mini_conversation_demo](https://github.com/pollen-robotics/reachy_mini_conversation_demo).

You can find more awesome apps and demos for Reachy Mini on [Hugging Face spaces](https://huggingface.co/spaces?q=reachy_mini)!

## Using the Python SDK

The API is designed to be simple and intuitive. You can control the robot's features such as the head, antennas, camera, speakers, and microphone. For instance, to move the head of the robot, you can use the `goto_target` method as shown in the example below:

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as reachy_mini:
    # Move the head up (10mm on z-axis) and roll it 15 degrees
    pose = create_head_pose(z=10, roll=15, degrees=True, mm=True)
    reachy_mini.goto_target(head=pose, duration=2.0)

    # Reset to default pose
    pose = create_head_pose() 
    reachy_mini.goto_target(head=pose, duration=2.0)
```

For a full description of the SDK, please refer to the [Python SDK documentation](./docs/source/python_sdk.mdx).

## Using the REST API

The daemon also provides a REST API via [fastapi](https://fastapi.tiangolo.com/) that you can use to control the robot and get its state. The API is accessible via HTTP and WebSocket.

By default, the API server runs on `http://localhost:8000`. The API is documented using OpenAPI, and you can access the documentation at `http://localhost:8000/docs` when the daemon is running.

More information about the API can be found in the [HTTP API documentation](./docs/source/rest_api.mdx).

## Share your apps with the commmunity

You can share your github repositories on social media or use [this guide](https://huggingface.co/blog/pollen-robotics/make-and-publish-your-reachy-mini-apps) to share your app even with users who can't code.

## Open source & contribution

This project is actively developed and maintained by the [Pollen Robotics team](https://www.pollen-robotics.com) and the [Hugging Face team](https://huggingface.co/). 

We welcome contributions from the community! If you want to report a bug or request a feature, please open an issue on GitHub. If you want to contribute code, please fork the repository and submit a pull request.

### 3D models

TODO

### Contributing

Development tools are available in the optional dependencies.

```bash
pip install -e .[dev]
pre-commit install
```

Your files will be checked before any commit. Checks may also be manually run with

```bash
pre-commit run --all-files
```

Checks are performed by Ruff. You may want to [configure your IDE to support it](https://docs.astral.sh/ruff/editors/setup/).

## Troubleshooting

see [dedicated section](docs/source/troubleshooting.mdx)

* **Join the Community:** We use [Discord](https://discord.gg/2bAhWfXme9) to share our moments with Reachy, build apps together, and get help.
* **Found a bug?** Open an issue on this repository.
* **Created an App?** [Share it with the community](https://huggingface.co/blog/pollen-robotics/make-and-publish-your-reachy-mini-apps).


## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
Hardware design files are licensed under Creative Commons BY-SA-NC.
