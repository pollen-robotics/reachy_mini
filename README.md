# Reachy Mini

## Installation

```bash
pip install -e .
```


> TODO: Outdated. Update the readme

## Usage

Look at the examples

Dance usage:

```
python groove_test.py --mode dance --dance_name head_tilt_roll --duration 20 --bpm 124

```

## Client / server

To run the mujoco server:

```bash
reachy-mini-simulation
```

To run the robot server (runs on the real robot):

```bash
reachy-mini
```

To use the client :

```python
from stewart_little_control import Client
client = Client(ip)
pose = np.eye(4)
client.send_pose(pose)
```
### Video client

MuJoCo publishes the camera stream at this address: "udp://@127.0.0.1:5005".
OpenCV can directly read this stream, as illustrated in the example below:

```python
python examples/video_client.py
```
Any UDP client should be able to read this stream:
```bash
ffplay -fflags nobuffer udp://127.0.0.1:5005
```

With the real robot, the camera is directly accessible with the USB connection, and can be directly read with OpenCV:

```python
import cv2
cap = cv2.VideoCapture(0)
...
```


## Simulation options

To choose a different scene:

```bash
reachy-mini-simulation -s minimal
```

### Simulation model used
https://polyhaven.com/a/food_apple_01
https://polyhaven.com/a/croissant
https://polyhaven.com/a/wooden_table_02
https://polyhaven.com/a/rubber_duck_toy
