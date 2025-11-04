# Reachy Mini

> ⚠️ Reachy Mini is still in beta. Expect bugs, some of them we won't fix right away if they are not a priority.

[Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) is an expressive, open-source robot designed for human-robot interaction, creative coding, and AI experimentation. We made it to be affordable, easy to use, hackable and cute, so that you can focus on building cool AI applications!

[![Reachy Mini Hello](/docs/assets/reachy_mini_hello.gif)](https://www.pollen-robotics.com/reachy-mini/)

### Versions Lite & Wireless

Reachy Mini's hardware comes in two flavors:
- **Reachy Mini lite**: where the robot is directly connected to your computer via USB. And the code that controls the robot (the daemon) runs on your computer.
- **Reachy Mini wireless**: where an Raspberry Pi is embedded in the robot, and the code that controls the robot (the daemon) runs on the Raspberry Pi. You can connect to it via Wi-Fi from your computer. (TODO: add link to section on how to set it up)

There is also a simulated version of Reachy Mini in [MuJoCo](https://mujoco.org) that you can use to prototype your applications before deploying them on the real robot. It behaves like the lite version where the daemon runs on your computer.

## Assembly guide

Follow our step-by-step [Assembly Guide](https://www.pollen-robotics.com/wp-content/uploads/2025/10/Reachy_Mini_Assembly_BETA_v2_LOW-compresse.pdf).
Most builders finish in about 3 hours, our current speed record is 43 minutes. The guide walks you through every step with clear visuals so you can assemble Reachy Mini confidently from start to finish. Enjoy the build!

## Software overview

This repository provides everything you need to control Reachy Mini, both in simulation and on the real robot. It consists of two main parts:

- **The 😈 Daemon 😈**: A background service that manages communication with the robot's motors and sensors, or with the simulation environment. It should be running before you can control the robot. It can run either for the simulation (MuJoCo) or for the real robot. 
- **🐍 SDK & 🕸️ API** to control the robot's main features (head, antennas, camera, speakers, microphone, etc.) and connect with your AI experimentation. Depending on your preferences and needs, there is a [Python SDK](#using-the-python-sdk) and a [HTTP REST API](#using-the-rest-api).

Using the [Python SDK](#using-the-python-sdk), making your robot move only require a few lines of code, as illustrated in the example below:

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

and using the [REST API](#using-the-rest-api), reading the current state of the robot:

```bash
curl 'http://localhost:8000/api/state/full'
```

Those two examples above assume that the daemon is already running (either in simulation or on the real robot) locally.

## Installation of the daemon and Python SDK

As mentioned above, before being able to use the robot, you need to run the daemon that will handle the communication with the motors.

We support and test on Linux and macOS. It's also working on Windows, but it is less tested at the moment. Do not hesitate to open an issue if you encounter any problem. 

The daemon is built in Python, so you need to have Python installed on your computer (versions from 3.10 to 3.13 are supported). We recommend using a virtual environment to avoid dependency conflicts with your other Python projects.

You can install Reachy Mini from the source code or from PyPI.

First, make sure `git-lfs` is installed on your system:

- On Linux: `sudo apt install git-lfs`
- On macOS: `brew install git-lfs`
- On Windows: [Follow the instructions here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=windows)

From PyPI, you can install the package with:

```bash
pip install reachy-mini
```

From the source code, you can install the package with:

```bash
git clone https://github.com/pollen-robotics/reachy_mini
pip install -e ./reachy_mini
```

The same package provides both the daemon and the Python SDK.

## Run the reachy mini daemon

Before being able to use the robot, you need to run the daemon that will handle the communication with the motors. This daemon can run either in simulation (MuJoCo) or on the real robot.

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

### For the lite version (connected via USB)

It should automatically detect the serial port of the robot. If it does not, you can specify it manually with the `-p` option:

```bash
reachy-mini-daemon -p <serial_port>
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

For a full description of the SDK, please refer to the [Python SDK documentation](./docs/python-sdk.md).

## Using the REST API

The daemon also provides a REST API via [fastapi](https://fastapi.tiangolo.com/) that you can use to control the robot and get its state. The API is accessible via HTTP and WebSocket.

By default, the API server runs on `http://localhost:8000`. The API is documented using OpenAPI, and you can access the documentation at `http://localhost:8000/docs` when the daemon is running.

More information about the API can be found in the [HTTP API documentation](./docs/rest-api.md).

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

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

The robot design files are licensed under the [TODO](TODO) license.

### Simulation model used

- https://polyhaven.com/a/food_apple_01
- https://polyhaven.com/a/croissant
- https://polyhaven.com/a/wooden_table_02
- https://polyhaven.com/a/rubber_duck_toy
