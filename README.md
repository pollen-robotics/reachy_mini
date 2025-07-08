# Reachy Mini

## Installation

```bash
pip install -e .
```

It requires Python 3.8 or later.

## Run the reachy mini daemon

Additional argument for both simulation and real robot:

```bash
--localhost-only: (default behavior). The server will only accept connections from localhost.
```

or

```bash
--no-localhost-only: If set, the server will accept connections from any connection on the local network.
```

### In simulation (MuJoCo)

```bash
reachy-mini-daemon --sim
```

Additional arguments:

```bash
--scene <empty|minimal> : (Default empty). Choose between a basic empty scene, or a scene with a table and some objects.
```

### On the real robot

```bash
reachy-mini-daemon
```

This should automatically detect the serial port of the robot. If it does not, you can specify it manually with the `-p` option:

```bash
reachy-mini-daemon -p <serial_port>
```

## Log with rerun

For both simulation and real robot, events can be logged with rerun. With the following command, rerun viewer will automaticall popup:

```bash
reachy-mini-daemon --rerun-mode [none|all|no-video]
```
Note that *all* locks the access to the camera which cannot be used by a client.

rerun needs to be installed: `pip install -e .[rerun]`


## Run the examples

Once the daemon is running, you can run the examples:

```bash
python examples/client_example.py
python examples/sequence.py
python examples/mirror_xyroll.py
python examples/head_track_demo.py
python examples/head_track_demo_rerun.py # reachy-mini-daemon --rerun-mode no-video
python examples/hand_track_demo.py
```

## To use the client :

```python
from reachy_mini import ReachyMini

with ReachyMini() as reachy_mini:
    ...
    ...
    reachy_mini.set_target(head=pose, antennas=antennas)
```

With the real robot, the camera is directly accessible with the USB connection, and can be directly read with OpenCV:

```python
import cv2
cap = cv2.VideoCapture(0)
...
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
