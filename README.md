# Reachy Mini

[Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) is the first open-source desktop robot designed to explore human-robot interaction and creative custom applications. We made it to be affordable, easy to use, hackable and cute, so that you can focus on build cool AI applications!

[![Reachy Mini Hello](/docs/assets/reachy_mini_hello.gif)](https://www.pollen-robotics.com/reachy-mini/)

This repository provides everything you need to control Reachy Mini, both in simulation and on the real robot. It consists of two main parts:

- **Daemon**: A background service that manages communication with the robot's motors and sensors, or with the simulation environment. 
- **Python API**: A simple to use API to control the robot's features (head, antennas, camera, speakers, microphone, etc.) from your own Python scripts that you can connect with your AI experimentation.

Making your robot move should only require a few lines of code, as illustrated in the example below:

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

## Installation

You can install Reachy Mini from the source code or from PyPI.

From PyPI, you can install the package with:

```bash
pip install reachy-mini
```

From the source code, you can install the package with:

```bash
git clone https://github.com/pollen-robotics/reachy_mini
pip install -e ./reachy_mini
```

It requires Python 3.8 or later.

## Run the reachy mini daemon

Before being able to use the robot, you need to run the daemon that will handle the communication with the motors. This daemon can run either in simulation (MuJoCo) or on the real robot.

```bash
reachy-mini-daemon
```

or run it via the Python module:

```bash
python -m reachy_mini.io.daemon
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

```bash
reachy-mini-daemon --sim
```

Additional arguments:

```bash
--scene <empty|minimal> : (Default empty). Choose between a basic empty scene, or a scene with a table and some objects.
```

<img src="https://www.pollen-robotics.com/wp-content/uploads/2025/06/Reachy_mini_simulation.gif" height="250" alt="Reachy Mini in MuJoCo">


### On the real robot

It should automatically detect the serial port of the robot. If it does not, you can specify it manually with the `-p` option:

```bash
reachy-mini-daemon -p <serial_port>
```

## Run the examples

Once the daemon is running, you can run the examples.

* To show the camera feed of the robot in a window, you can run:

    ```bash
    python examples/camera_viewer.py
    ```

* To show an example on how to use the look_at method to make the robot look at a point in 2D space, you can run:

    ```bash
    python examples/look_at_image.py
    ```

* To illustrate the differences between the interpolation methods when running the goto method (linear, minimum jerk, cartoon, etc).

    ```bash
    python examples/goto_interpolation_playground.py
    ```

## To use the API:

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

With the real robot, the camera is directly accessible with the USB connection, and can be directly read with OpenCV:

```python
import cv2

from reachy_mini.io.cam_utils import find_camera

cap = find_camera()
while True:
    success, frame = cap.read()
    if success:
        cv2.imshow("Reachy Mini Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

If you know the camera id on OpenCV, you can also directly use it:

```python
import cv2

cap = cv2.VideoCapture(0)  # Replace 0 with your camera ID
```

---------

### Video client (TODO (removed from this release for performance reasons))

MuJoCo publishes the camera stream at this address: "udp://@127.0.0.1:5005".
OpenCV can directly read this stream, as illustrated in the example below:

```python
python examples/video_client.py
```

Any UDP client should be able to read this stream:

```bash
ffplay -fflags nobuffer udp://127.0.0.1:5005
```

### Simulation model used

https://polyhaven.com/a/food_apple_01

https://polyhaven.com/a/croissant

https://polyhaven.com/a/wooden_table_02

https://polyhaven.com/a/rubber_duck_toy

## Contribute

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
